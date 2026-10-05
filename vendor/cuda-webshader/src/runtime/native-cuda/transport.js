// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ./LICENSE.
// Vendored CUDA transport; provenance and local changes are in ./NOTICE.md.
// API shape follows SamG-Coder/cuda-webshader's GpuRuntime. Native backend.
const MAX_TRANSFER = 512 * 1024;
const constructors = [Float32Array, Uint32Array, Int32Array, Uint8Array, Uint16Array];
const dimensions = value => {
  if (!Array.isArray(value) || value.length < 1 || value.length > 3 ||
      value.some(x => !Number.isInteger(x) || x < 1 || x > 0xffffffff))
    throw new RangeError('Expected one to three positive grid/block dimensions.');
  return [...value, 1, 1].slice(0, 3);
};
function encode(data) {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  if (bytes.byteLength > MAX_TRANSFER) throw new RangeError('Transfer exceeds 512 KiB.');
  let binary = '';
  for (let i = 0; i < bytes.length; i += 8192)
    binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(binary);
}
function decode(text) {
  const binary = atob(text), result = new Uint8Array(binary.length);
  for (let i = 0; i < result.length; ++i) result[i] = binary.charCodeAt(i);
  return result;
}
function signature(source, entry, explicit) {
  if (explicit) {
    if (!Array.isArray(explicit) || explicit.length > 32 || explicit.some(p =>
      !p || !/^[A-Za-z_]\w*$/.test(p.name) || !['buffer', 'i32', 'u32', 'f32'].includes(p.type)))
      throw new TypeError('parameters must be ordered {name, type} descriptors.');
    if (new Set(explicit.map(p => p.name)).size !== explicit.length) throw new TypeError('Duplicate parameter name.');
    return explicit.map(p => ({...p}));
  }
  if (!/^[A-Za-z_]\w*$/.test(entry))
    throw new TypeError('Use explicit parameters for template or qualified entry points.');
  // Inspect only a conventional entry signature. NVRTC compiles all CUDA
  // syntax; advanced signatures/macros use explicit ABI descriptors.
  const clean = source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, ' ');
  const match = new RegExp(`__global__\\s+void\\s+${entry}\\s*\\(([^()]*)\\)`).exec(clean);
  if (!match) throw new TypeError('Entry signature needs explicit parameters.');
  if (!match[1].trim() || match[1].trim() === 'void') return [];
  return match[1].split(',').map(parameter => {
    const tokens = parameter.trim().replace(/\b(const|volatile|__restrict__|__restrict|restrict)\b/g, ' ').trim();
    const match = /^(.+?)\s*\b([A-Za-z_]\w*)\s*$/.exec(tokens);
    if (!match) throw new TypeError('Use explicit parameters for this CUDA signature.');
    const type = match[1].replace(/\s+/g, ' ').trim();
    const scalar = {float: 'f32', int: 'i32', 'signed int': 'i32', unsigned: 'u32', 'unsigned int': 'u32', uint32_t: 'u32', int32_t: 'i32'};
    const mapped = type.includes('*') ? 'buffer' : scalar[type];
    if (!mapped || type.includes('&') || type.includes('['))
      throw new TypeError(`Unsupported parameter ${parameter}; use explicit supported ABI descriptors.`);
    return {name: match[2], type: mapped};
  });
}
export async function SupportsNativeCuda() {
  return !!(await globalThis.navigator?.cuda?.SupportsNativeCuda?.());
}
export async function requestPermission() {
  if (!globalThis.navigator?.cuda) throw new DOMException('This browser has no native CUDA API.', 'NotSupportedError');
  return navigator.cuda.requestPermission();
}
export class GpuRuntime {
  static SupportsNativeCuda = SupportsNativeCuda;
  static requestPermission = requestPermission;
  static async create({backend = 'native'} = {}) {
    if (backend !== 'native') throw new TypeError('This module implements the native backend.');
    if (!await SupportsNativeCuda()) throw new DOMException('Native CUDA is unavailable.', 'NotSupportedError');
    const state = await navigator.cuda.queryPermission();
    if (state !== 'granted') throw new DOMException('Call requestPermission() from a user click before create().', 'NotAllowedError');
    const {session} = JSON.parse(await navigator.cuda.execute('cuda.open','{}'));
    return new GpuRuntime(navigator.cuda,session);
  }
  constructor(transport,session) {
    this.transport = transport;
    this.session = session;
    this.backend = 'native-cuda'; this.disposed = false; this.lost = null;
    this.queue = Promise.resolve(); this.buffers = new Set(); this.pipelineCache = new Map();
  }
  assertAlive() {
    if (this.disposed) throw new Error('Runtime is disposed.');
    if (this.lost) throw this.lost;
  }
  enqueue(operation, payload, fatal = true) {
    this.assertAlive();
    const result = this.queue.then(async () => JSON.parse(await this.transport.execute(operation, JSON.stringify({...payload,$session:this.session}))));
    this.queue = fatal ? result : result.catch(() => {});
    // Retain errors for idle/read while avoiding unhandled rejections from
    // synchronous API methods such as createBuffer/write/submit.
    result.catch(error => { if (fatal) this.lost ||= error; });
    return result;
  }
  checkResource(resource) {
    this.assertAlive();
    if (!resource || resource.runtime !== this || resource.destroyed) throw new TypeError('Buffer is destroyed or belongs to another runtime.');
  }
  createBuffer(dataOrBytes, {label = 'compute buffer'} = {}) {
    this.assertAlive();
    const data = ArrayBuffer.isView(dataOrBytes) ? dataOrBytes : null;
    const byteLength = data ? data.byteLength : dataOrBytes;
    if (!Number.isSafeInteger(byteLength) || byteLength < 0 || byteLength > 64 * 1024 * 1024)
      throw new RangeError('Buffer size must be 0..64 MiB.');
    const size = Math.max(4, Math.ceil(byteLength / 4) * 4);
    const resource = {runtime: this, byteLength, size, label, destroyed: false, id: null};
    const bytes = data && new Uint8Array(data.buffer,data.byteOffset,data.byteLength);
    resource.ready = this.enqueue('cuda.createBuffer', {byteLength: size, ...(bytes ? {data: encode(bytes.subarray(0,MAX_TRANSFER))} : {})})
      .then(result => { resource.id = result.id; });
    resource.ready.catch(() => {}); this.buffers.add(resource);
    if (bytes) for (let offset = MAX_TRANSFER; offset < byteLength; offset += MAX_TRANSFER) {
      const data = encode(bytes.subarray(offset,offset+MAX_TRANSFER));
      this.enqueueDeferred('cuda.write',async () => { await resource.ready; return {id:resource.id,offset,data}; });
    }
    return resource;
  }
  write(resource, data, offset = 0) {
    this.checkResource(resource);
    if (!ArrayBuffer.isView(data) || !Number.isInteger(offset) || offset < 0 || offset % 4 ||
        data.byteLength % 4 || offset + data.byteLength > resource.size) throw new RangeError('Write must be aligned and fit the buffer.');
    const bytes = new Uint8Array(data.buffer,data.byteOffset,data.byteLength);
    for (let start = 0; start < bytes.byteLength; start += MAX_TRANSFER) {
      const encoded = encode(bytes.subarray(start,start+MAX_TRANSFER)); // Snapshot at call time.
      this.enqueueDeferred('cuda.write', async () => { await resource.ready; return {id: resource.id, offset:offset+start, data: encoded}; });
    }
  }
  enqueueDeferred(operation, payload) {
    this.assertAlive();
    const result = this.queue.then(async () => JSON.parse(await this.transport.execute(operation, JSON.stringify({...await payload(),$session:this.session}))));
    this.queue = result; result.catch(error => { this.lost ||= error; }); return result;
  }
  async read(resource, Type = Float32Array, byteLength = resource.byteLength, offset = 0) {
    this.checkResource(resource);
    if (!constructors.includes(Type) || !Number.isInteger(byteLength) || byteLength < 0 ||
        !Number.isInteger(offset) || offset < 0 || offset + byteLength > resource.size || byteLength % Type.BYTES_PER_ELEMENT)
      throw new RangeError('Invalid readback type or range.');
    const reads = [];
    for (let start = 0; start < byteLength; start += MAX_TRANSFER) {
      const size = Math.min(MAX_TRANSFER,byteLength-start);
      reads.push(this.enqueueDeferred('cuda.read',async () => {
        await resource.ready; return {id:resource.id,byteLength:size,offset:offset+start};
      }).then(result => {
        const bytes = decode(result.data);
        if (bytes.byteLength !== size) throw new Error('Native readback length mismatch.');
        return bytes;
      }));
    }
    // Queue all chunks before yielding, so later writes cannot split this read.
    const chunks = await Promise.all(reads), bytes = new Uint8Array(byteLength);
    chunks.forEach((chunk,i) => bytes.set(chunk,i*MAX_TRANSFER));
    return new Type(bytes.buffer);
  }
  async kernel(source, {entry, workgroupSize = [128, 1, 1], parameters, sharedMemoryBytes = 0} = {}) {
    this.assertAlive();
    if (typeof source !== 'string' || !source.length || source.length > 262144) throw new TypeError('kernel requires CUDA source text.');
    if (typeof entry !== 'string') throw new TypeError('Specify a CUDA entry point.');
    const block = dimensions(workgroupSize), params = signature(source, entry, parameters);
    if (!Number.isInteger(sharedMemoryBytes) || sharedMemoryBytes < 0 || sharedMemoryBytes > 49152) throw new RangeError('Invalid shared memory size.');
    const key = JSON.stringify([source,entry,block,params,sharedMemoryBytes]);
    if (!this.pipelineCache.has(key)) {
      const promise = this.enqueue('cuda.kernel', {source,entry},false).then(result => new Kernel(this,result.id,params,block,sharedMemoryBytes));
      this.pipelineCache.set(key,promise); promise.catch(() => this.pipelineCache.delete(key));
    }
    return this.pipelineCache.get(key);
  }
  batch() { this.assertAlive(); return new ComputeBatch(this); }
  async idle() { this.assertAlive(); await this.queue; }
  destroyBuffer(resource) {
    this.checkResource(resource); resource.destroyed = true; this.buffers.delete(resource);
    this.enqueueDeferred('cuda.destroyBuffer', async () => { await resource.ready; return {id: resource.id}; });
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of this.buffers) buffer.destroyed = true;
    this.buffers.clear(); this.pipelineCache.clear();
    // After successful queue retirement, close this document's native session.
    // Also close on failure so device resources cannot linger after a rejection.
    this.queue.finally(() => this.transport.close()).catch(() => {});
  }
}
export class Kernel {
  constructor(runtime, id, parameters, block, sharedMemoryBytes) { Object.assign(this,{runtime,id,parameters,block,sharedMemoryBytes}); }
  bind(buffers, scalars = {}) { return new Invocation(this,buffers,scalars); }
}
export class Invocation {
  constructor(kernel,buffers,scalars) {
    this.kernel = kernel; this.buffers = {...buffers}; this.scalars = {...scalars};
    const names = new Set(kernel.parameters.map(p => p.name));
    for (const name of [...Object.keys(buffers),...Object.keys(scalars)]) if (!names.has(name)) throw new TypeError(`Unknown argument ${name}.`);
    for (const param of kernel.parameters) {
      if (param.type === 'buffer') kernel.runtime.checkResource(this.buffers[param.name]);
      else if (!Object.hasOwn(scalars,param.name)) throw new TypeError(`Missing scalar ${param.name}.`);
    }
  }
  setScalars(values) { Object.assign(this.scalars, values); return this; }
}
export class ComputeBatch {
  constructor(runtime) { this.runtime = runtime; this.commands = []; this.ended = false; }
  dispatch(invocation, workgroups) {
    if (this.ended) throw new Error('Batch already submitted or discarded.');
    if (invocation.kernel.runtime !== this.runtime) throw new TypeError('Kernel belongs to another runtime.');
    if (this.commands.length >= 256) throw new RangeError('Batch exceeds 256 dispatches.');
    const kernel = invocation.kernel, grid = dimensions(workgroups);
    // Snapshot scalars at dispatch, not at submit. Buffers remain GPU-resident.
    const args = kernel.parameters.map(param => {
      if (param.type === 'buffer') { const resource = invocation.buffers[param.name]; this.runtime.checkResource(resource); return {resource}; }
      const value = invocation.scalars[param.name];
      if (!Number.isFinite(value) || (param.type !== 'f32' && !Number.isInteger(value)) ||
          (param.type === 'u32' && (value < 0 || value > 0xffffffff)) ||
          (param.type === 'i32' && (value < -0x80000000 || value > 0x7fffffff))) throw new RangeError(`Invalid scalar ${param.name}.`);
      return {type: param.type, value};
    });
    this.commands.push({kernel: kernel.id, grid, block: kernel.block, sharedMemoryBytes: kernel.sharedMemoryBytes, args});
    return this;
  }
  submit() {
    if (this.ended) throw new Error('Batch already ended.'); this.ended = true;
    if (!this.commands.length) return this;
    const commands = this.commands;
    this.runtime.enqueueDeferred('cuda.dispatch', async () => {
      const jobs = [];
      for (const {args,...command} of commands) {
        const arguments_ = [];
        for (const arg of args) {
          if (arg.resource) { await arg.resource.ready; arguments_.push({buffer: arg.resource.id}); }
          else arguments_.push(arg);
        }
        jobs.push({...command,arguments: arguments_});
      }
      return {jobs};
    });
    return this;
  }
  discard() { if (this.ended) throw new Error('Batch already ended.'); this.ended = true; this.commands = []; }
}
