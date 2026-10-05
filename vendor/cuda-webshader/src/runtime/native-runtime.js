import {GpuRuntime as NativeTransport,ComputeBatch as NativeBatch} from './native-cuda/transport.js';

let owner=null,opening=false;
const unsupported=name=>new DOMException(`${name} requires backend: 'webgpu'. Native CUDA currently supports buffer compute, not WebGPU resource interoperability.`,'NotSupportedError');
const types={i32:v=>Number.isInteger(v)&&v>=-2147483648&&v<=2147483647,u32:v=>Number.isInteger(v)&&v>=0&&v<=4294967295,f32:v=>typeof v==='number'&&Number.isFinite(v)&&Number.isFinite(Math.fround(v))};
function validateScalars(parameters,values){
  for(const [name,value] of Object.entries(values)){
    const param=parameters.find(p=>p.name===name&&p.type!=='buffer');
    if(!param)throw new TypeError(`Unknown scalar parameter '${name}'.`);
    if(!types[param.type](value))throw new RangeError(`Invalid scalar ${name}.`);
  }
}
export class NativeRuntime extends NativeTransport {
  static async open(api,options){
    if(owner||opening)throw new DOMException('One native CUDA runtime may own this document. Dispose it before creating another.','InvalidStateError');
    opening=true;
    try {const {session}=JSON.parse(await api.execute('cuda.open','{}'));return owner=new NativeRuntime(api,session,options);}
    finally{opening=false;}
  }
  constructor(api,session,options={}){
    super(api,session);this.nativeStatus=options.nativeStatus;this.onError=options.onError||(()=>{});this.reportedLoss=false;
    this.stats={pipelineCompiles:0,pipelineCacheHits:0,bindGroupsCreated:0,submissions:0,dispatches:0,uniformBytesUploaded:0,dataBytesUploaded:0,readbackBytes:0};
    this.nativeKernels=new WeakMap();
  }
  describe(){return {backend:this.backend,native:this.nativeStatus,vendor:'NVIDIA',architecture:'',device:'',description:'ChromiumRTXCuda native CUDA',features:['native-cuda'],timestampQuery:false,limits:{maxBufferSize:67108864,maxSessionBufferBytes:67108864,maxBuffers:256,maxModules:128,maxDispatchesPerBatch:256,maxSharedMemoryBytes:49152},webgpuInterop:false};}
  observe(promise,fatal){promise.catch(error=>{if(fatal&&!this.reportedLoss){this.reportedLoss=true;this.onError(error);}});return promise;}
  enqueue(operation,payload,fatal=true){return this.observe(super.enqueue(operation,payload,fatal),fatal);}
  enqueueDeferred(operation,payload){return this.observe(super.enqueueDeferred(operation,payload),true);}
  createBuffer(data,options={}){
    if(options.usage)throw unsupported('GPUBuffer usage flags');
    const size=Math.max(4,Math.ceil((ArrayBuffer.isView(data)?data.byteLength:data)/4)*4);
    if(this.buffers.size>=256||size+[...this.buffers].reduce((n,b)=>n+b.size,0)>67108864)throw new RangeError('Native CUDA session exceeds 256 buffers or 64 MiB.');
    const buffer=super.createBuffer(data,options);buffer.owned=true;this.stats.dataBytesUploaded+=ArrayBuffer.isView(data)?data.byteLength:0;return buffer;
  }
  write(buffer,data,offset=0){super.write(buffer,data,offset);this.stats.dataBytesUploaded+=data.byteLength;}
  async read(buffer,Type=Float32Array,byteLength=buffer.byteLength,offset=0){const data=await super.read(buffer,Type,byteLength,offset);this.stats.readbackBytes+=byteLength;return data;}
  async kernel(sourceOrArtifact,options={}){
    let source=sourceOrArtifact,defaults={};
    if(typeof source!=='string'){
      const native=source?.native;
      if(!native||native.version!==1)throw new TypeError('Native CUDA requires CUDA source text or a dual-backend artifact. Recompile with includeNativeSource: true; WGSL-only artifacts cannot execute as CUDA.');
      source=native.source;defaults=native.defaults||{};
      options={entry:native.entry,workgroupSize:native.workgroupSize,parameters:native.parameters,sharedMemoryBytes:native.sharedMemoryBytes,...options};
    }else if(!options.entry){
      const clean=source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g,' '),entries=[...clean.matchAll(/__global__\s+void\s+([A-Za-z_]\w*)\s*\(/g)];
      if(entries.length===1)options={...options,entry:entries[0][1]};
    }
    if(options.objectArena||options.scheduleDeviceLaunches||options.valueBuffers||options.bufferAliases)throw unsupported('Compiler-managed resources');
    const block=options.workgroupSize??[128,1,1];if(block.reduce((a,b)=>a*b,1)>1024)throw new RangeError('Native CUDA block exceeds 1024 threads.');
    const kernel=await super.kernel(source,options);
    let variants=this.nativeKernels.get(kernel);const variant=JSON.stringify(defaults);
    if(variants)this.stats.pipelineCacheHits++;else{this.stats.pipelineCompiles++;this.nativeKernels.set(kernel,variants=new Map());}
    let wrapped=variants.get(variant);if(wrapped)return wrapped;
    wrapped={runtime:this,metadata:{workgroupSize:kernel.block},bind:(buffers,scalars={},bindOptions={})=>{
      if(Object.keys(bindOptions).length)throw unsupported('WebGPU binding options');
      const values={...defaults,...scalars};validateScalars(kernel.parameters,values);
      for(const name of Object.keys(buffers))if(!kernel.parameters.some(p=>p.name===name&&p.type==='buffer'))throw new TypeError(`Unknown buffer parameter '${name}'.`);
      const invocation=kernel.bind(buffers,values);
      invocation.setScalars=updates=>{validateScalars(kernel.parameters,updates);Object.assign(invocation.scalars,updates);return invocation;};
      return invocation;
    }};
    variants.set(variant,wrapped);return wrapped;
  }
  batch(options={}){this.assertAlive();if(Object.keys(options).length)throw unsupported('WebGPU batch options');return new ComputeBatch(this);}
  dispose(){
    if(this.disposed)return this.closed;
    this.disposed=true;for(const b of this.buffers)b.destroyed=true;this.buffers.clear();this.pipelineCache.clear();
    // Keep ownership through close, so an older asynchronous dispose cannot
    // close a newly opened document session.
    this.closed=this.queue.catch(()=>{}).then(()=>this.transport.close()).finally(()=>{if(owner===this)owner=null;});
    this.closed.catch(error=>this.onError(error));return this.closed;
  }
}
class ComputeBatch extends NativeBatch {
  dispatch(invocation,workgroups){
    this.runtime.assertAlive();
    validateScalars(invocation.kernel.parameters,invocation.scalars);
    return super.dispatch(invocation,workgroups);
  }
  submit(){
    this.runtime.assertAlive();
    // A buffer destroyed after recording must not reach the driver.
    for(const command of this.commands)for(const arg of command.args)if(arg.resource)this.runtime.checkResource(arg.resource);
    const result=super.submit();if(this.commands.length){this.runtime.stats.submissions++;this.runtime.stats.dispatches+=this.commands.length;}return result;
  }
  copy(){throw unsupported('GPU buffer copies');}
  toTexture(){throw unsupported('GPU texture presentation');}
}
for(const name of ['importBuffer','createTexture1D','createTexture2D','createTexture3D','createCubemapTexture','createLayeredTexture2D','createByteTexture2D','createUintTexture2D','createLinearTexture','destroyTexture','readTextureSlice','createObjectArena','sortPairs','inverseFFT2D','complexFFT2D','realFFT2D','exclusiveScan'])
  NativeRuntime.prototype[name]=function(){throw unsupported(name);};
