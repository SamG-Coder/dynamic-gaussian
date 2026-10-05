// Browser-managed CUDA / WebGPU interoperability. Native handles never enter JS.
import {createAccelerationStructure,createRayTracingPipeline,buildAccelerationStructure,traceRays} from './native-optix.js';
const unavailable = reason => ({version:1,available:false,sharedBuffers:false,sharedTextures:false,
  samePhysicalGpu:false,textureFormats:[],synchronization:'unavailable',reason});
export async function getNativeInteropCapabilities(device) {
  const api=globalThis.navigator?.cuda;
  if(!api?.getInteropCapabilities)return unavailable('Browser has no native CUDA / WebGPU sharing API.');
  let permission='prompt';
  try {
    permission=await api.queryPermission();
    if(permission!=='granted')return {...unavailable('Native GPU permission is '+permission),permission};
    const capabilities=JSON.parse(await api.getInteropCapabilities(device));
    return {...capabilities,canvasPresentation:typeof api.createCanvasSurface==='function',permission,available:capabilities.samePhysicalGpu&&capabilities.sharedBuffers&&
      capabilities.synchronization==='d3d12-fence-cuda-external-semaphore'};
  } catch(error) {return {...unavailable(error.message),permission};}
}
export function supportsInteropRequirements(capabilities,requirements={}) {
  const c=capabilities,r=requirements;
  if(!c.available||!c.samePhysicalGpu||!c.sharedBuffers||c.synchronization!=='d3d12-fence-cuda-external-semaphore')return false;
  if(r.optix&&!c.optix?.available)return false;
  if(r.nativeOwnedBuffers&&!c.nativeOwnedBuffers)return false;
  if(r.canvasPresentation&&!c.canvasPresentation)return false;
  if((r.sharedTextures||r.textureFormats?.length)&&(!c.sharedTextures||!(r.textureFormats||[]).every(f=>c.textureFormats.includes(f))))return false;
  if(r.gpuBufferToTexture&&!c.gpuBufferToTexture)return false;
  for(const [key,limit] of [['maxResourceBytes','maxResourceBytes'],['sharedBytes','maxSharedBytes'],
    ['resources','maxResources'],['blocksPerLaunch','maxBlocksPerLaunch'],['textureDimension2D','maxTextureDimension2D']]) {
    const maximum=limit==='maxBlocksPerLaunch'&&c.maxGridDimensions?Math.min(Number.MAX_SAFE_INTEGER,c.maxGridDimensions.reduce((a,b)=>a*b,1)):c[limit];
    if(r[key]!==undefined&&(!Number.isSafeInteger(r[key])||r[key]<0||!Number.isSafeInteger(maximum)||r[key]>maximum))return false;
  }
  if(r.bufferUsage!==undefined&&(!Number.isInteger(r.bufferUsage)||(r.bufferUsage&~c.bufferUsageMask)))return false;
  if(r.textureUsage!==undefined&&(!Number.isInteger(r.textureUsage)||(r.textureUsage&~c.textureUsageMask)))return false;
  if(r.textureDimension!==undefined&&r.textureDimension!==c.textureDimension)return false;
  for(const key of ['mipLevelCount','sampleCount','textureArrayLayers'])if(r[key]!==undefined&&r[key]!==c[key])return false;
  return true;
}
const dimensions=value=>{
  const v=Array.isArray(value)?[...value]:[value];while(v.length<3)v.push(1);
  if(v.length!==3||v.some(n=>!Number.isInteger(n)||n<1||n>0xffffffff))throw new RangeError('Expected one to three positive grid/block dimensions.');
  return v;
};
const scalarTypes={f32:v=>typeof v==='number'&&Number.isFinite(v)&&Number.isFinite(Math.fround(v)),
  i32:v=>Number.isInteger(v)&&v>=-2147483648&&v<=2147483647,u32:v=>Number.isInteger(v)&&v>=0&&v<=4294967295};
function parameters(source,entry,explicit) {
  if(explicit) {
    if(!Array.isArray(explicit)||explicit.length>32||explicit.some(p=>!p||!/^\w+$/.test(p.name)||
      !['buffer','surface','f32','i32','u32'].includes(p.type))||new Set(explicit.map(p=>p.name)).size!==explicit.length)
      throw new TypeError('Invalid native interop parameter descriptors.');
    return explicit.map(p=>({...p}));
  }
  if(!/^[A-Za-z_]\w*$/.test(entry))throw new TypeError('Qualified entries require explicit parameters.');
  const clean=source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g,' '),match=new RegExp(`__global__\\s+void\\s+${entry}\\s*\\(([^()]*)\\)`).exec(clean);
  if(!match)throw new TypeError('Native entry signature needs explicit parameters.');
  if(!match[1].trim()||match[1].trim()==='void')return [];
  return match[1].split(',').map(p=>{
    const m=/^(.+?)\s*\b([A-Za-z_]\w*)\s*$/.exec(p.replace(/\b(const|volatile|__restrict__|__restrict|restrict)\b/g,' ').trim());
    if(!m)throw new TypeError('Unsupported native parameter signature.');
    const raw=m[1].replace(/\s+/g,' ').trim();
    const type=raw.includes('*')?'buffer':({float:'f32',int:'i32',unsigned:'u32','unsigned int':'u32',
      cudaSurfaceObject_t:'surface'})[raw];
    if(!type||raw.includes('&')||raw.includes('['))throw new TypeError('Unsupported native parameter '+p);
    return {name:m[2],type};
  });
}
let owner=null,opening=false;
export class NativeInterop {
  static async open(runtime,options={}) {
    if(owner||opening){runtime.nativeInteropStatus=unavailable('A native interoperability session already owns this document.');return null;}
    opening=true;
    try {
      const api=globalThis.navigator?.cuda;
      if(options.requestPermission&&api&&await api.queryPermission()==='prompt')await api.requestPermission();
      const capabilities=await getNativeInteropCapabilities(runtime.device);
      runtime.nativeInteropStatus=capabilities;
      if(!supportsInteropRequirements(capabilities,options.requirements))return null;
      const session=JSON.parse(await api.execute('cuda.open','{}')).session;
      return owner=new NativeInterop(runtime,api,session,capabilities);
    } catch(error) {runtime.nativeInteropStatus=unavailable(error.message);return null;}
    finally {opening=false;}
  }
  constructor(runtime,api,session,capabilities) {
    Object.assign(this,{runtime,api,session,capabilities});this.resources=new Set();this.kernels=new Map();
    this.optixObjects=new Set();this.deviceBuffers=new Set();this.deviceBytes=0;
    this.tail=Promise.resolve();this.closed=false;this.failure=null;
    this.stats={dispatches:0,submissions:0,sharedBytes:0};
    runtime.device.lost.then(()=>this.dispose());
  }
  assertAlive(){this.runtime.assertAlive();if(this.closed)throw new DOMException('Native interop session is closed.','InvalidStateError');if(this.failure)throw this.failure;}
  enqueue(work,{fatal=true}={}) {
    this.assertAlive();const result=this.tail.then(()=>{this.assertAlive();return work();});
    this.tail=result.catch(error=>{if(fatal)this.failure=error;});return result;
  }
  supports(requirements){return !this.closed&&!this.failure&&supportsInteropRequirements(this.capabilities,requirements);}
  check(resource,type) {
    this.runtime.checkResource(resource);
    if(type==='buffer'&&this.deviceBuffers.has(resource)&&resource.nativeBufferId)return;
    if(!this.resources.has(resource)||!resource.nativeResource?.id||
      (type==='buffer'?!resource.gpuBuffer:!(resource.gpuTexture||resource.nativeCanvas)))
      throw new TypeError('Native kernels require an explicitly created shared '+type+'.');
  }
  async createBuffer(dataOrBytes,{label='CUDA / WebGPU shared buffer',usage=0}={}) {
    const data=ArrayBuffer.isView(dataOrBytes)?dataOrBytes:null,bytes=data?data.byteLength:dataOrBytes;
    if(!Number.isSafeInteger(bytes)||bytes<0)throw new RangeError('Invalid shared buffer size.');
    const size=Math.max(4,Math.ceil(bytes/4)*4);
    const flags=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST|usage;
    if(size>this.runtime.device.limits.maxStorageBufferBindingSize||size>this.capabilities.maxResourceBytes||
      (flags&~this.capabilities.bufferUsageMask))throw new RangeError('Shared buffer requirements are unsupported.');
    const snapshot=data?new Uint8Array(size):null;
    if(snapshot)snapshot.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));
    return this.enqueue(async()=>{
      const native=await this.api.createSharedBuffer(this.runtime.device,size,flags);
      const resource=this.runtime.importBuffer(native.buffer,bytes,label);
      resource.nativeResource=native;resource.owned=true;this.runtime.buffers.add(resource);this.resources.add(resource);
      this.stats.sharedBytes+=size;
      if(snapshot)this.runtime.write(resource,snapshot);
      return resource;
    });
  }
  async createDeviceBuffer(byteLength,{label='CUDA-owned buffer'}={}) {
    this.assertAlive();
    if(!this.capabilities.nativeOwnedBuffers)throw new DOMException('Native-owned interop buffers require an updated ChromiumRTXCuda.','NotSupportedError');
    if(!Number.isSafeInteger(byteLength)||byteLength<4||byteLength%4)throw new RangeError('Native buffer size must be a positive multiple of four.');
    return this.enqueue(async()=>{
      if(this.deviceBuffers.size>=256||this.deviceBytes+byteLength>this.capabilities.maxNativeOwnedBytes)throw new RangeError('Native buffer allocation budget exceeded.');
      const {id}=JSON.parse(await this.api.execute('cuda.createBuffer',JSON.stringify({byteLength,$session:this.session})));
      const resource={runtime:this.runtime,nativeBufferId:id,byteLength,size:byteLength,label,destroyed:false};
      this.deviceBuffers.add(resource);this.deviceBytes+=byteLength;return resource;
    },{fatal:false});
  }
  async destroyDeviceBuffer(resource) {
    this.check(resource,'buffer');
    if(!this.deviceBuffers.has(resource))throw new TypeError('Expected a CUDA-owned buffer.');
    resource.destroyed=true;
    return this.enqueue(async()=>{
      await this.api.execute('cuda.idle',JSON.stringify({$session:this.session}));
      await this.api.execute('cuda.destroyBuffer',JSON.stringify({id:resource.nativeBufferId,$session:this.session}));
      this.deviceBuffers.delete(resource);this.deviceBytes-=resource.byteLength;
    });
  }
  async createTexture({width,height,format='rgba8unorm',usage=GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST|
    GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING,label='CUDA / WebGPU shared texture',
    dimension='2d',depthOrArrayLayers=1,mipLevelCount=1,sampleCount=1}={}) {
    if(!this.capabilities.sharedTextures||!this.capabilities.textureFormats.includes(format)||
      ![width,height].every(n=>Number.isInteger(n)&&n>0&&n<=Math.min(8192,this.runtime.device.limits.maxTextureDimension2D))||
      dimension!=='2d'||depthOrArrayLayers!==1||mipLevelCount!==1||sampleCount!==1||!usage||(usage&~31))
      throw new RangeError('Shared textures require a supported format, 2D extent, one layer/mip/sample, and supported usage.');
    return this.enqueue(async()=>{
      const native=await this.api.createSharedTexture(this.runtime.device,width,height,format,usage);
      const resource={id:'native-texture-'+native.id,runtime:this.runtime,owned:true,destroyed:false,nativeResource:native,
        gpuTexture:native.texture,view:native.texture.createView(),format,width,height,depth:1,dimension:'2d',
        storage:!!(usage&GPUTextureUsage.STORAGE_BINDING),label,size:width*height*({rgba8unorm:4,rgba16float:8,rgba32float:16,r32float:4}[format])};
      this.runtime.textures.add(resource);this.resources.add(resource);this.stats.sharedBytes+=resource.size;return resource;
    });
  }
  async createCanvasTarget(canvas,{buffers=3,context=canvas.getContext('webgpu')}={}) {
    this.assertAlive();
    if(!this.capabilities.canvasPresentation)throw new DOMException('Native canvas presentation requires an updated ChromiumRTXCuda.','NotSupportedError');
    if(!Number.isInteger(buffers)||buffers<2||buffers>4)throw new RangeError('A canvas target needs two to four surfaces.');
    const width=canvas.width,height=canvas.height;
    if(!context||![width,height].every(n=>Number.isInteger(n)&&n>0&&n<=Math.min(8192,this.runtime.device.limits.maxTextureDimension2D)))throw new RangeError('Invalid native canvas dimensions or context.');
    return this.enqueue(async()=>{
      const surfaces=[];
      try {
        for(let i=0;i<buffers;i++) {
          const native=await this.api.createCanvasSurface(this.runtime.device,width,height);
          const resource={runtime:this.runtime,nativeResource:native,nativeCanvas:true,
            width,height,format:'rgba8unorm',size:width*height*4,destroyed:false};
          this.resources.add(resource);this.stats.sharedBytes+=resource.size;surfaces.push(resource);
        }
      } catch(error) {
        for(const resource of surfaces){resource.nativeResource.destroy();this.resources.delete(resource);this.stats.sharedBytes-=resource.size;}
        throw error;
      }
      return new NativeCanvasTarget(this,canvas,context,surfaces);
    },{fatal:false});
  }
  async kernel(sourceOrArtifact,options={}) {
    this.assertAlive();
    const artifact=typeof sourceOrArtifact==='string'?null:sourceOrArtifact,native=artifact?.native;
    if(artifact&&(!native||native.version!==1))throw new TypeError('Native kernels require original CUDA or a dual-backend artifact.');
    const source=artifact?native.source:sourceOrArtifact;
    const {entry,workgroupSize=[128,1,1],parameters:explicit,sharedMemoryBytes=0}={...native,...options};
    if(typeof source!=='string'||!source.length||new TextEncoder().encode(source).length>262144)throw new RangeError('Native CUDA source must fit 256 KiB.');
    const block=dimensions(workgroupSize),params=parameters(source,entry,explicit);
    if(block.reduce((a,b)=>a*b,1)>1024||!Number.isInteger(sharedMemoryBytes)||sharedMemoryBytes<0||sharedMemoryBytes>49152)
      throw new RangeError('Unsupported CUDA block/shared memory size.');
    const key=JSON.stringify([source,entry,block,params,sharedMemoryBytes,native?.defaults||{}]);
    if(!this.kernels.has(key)) {
      const pending=this.enqueue(async()=>{
        const {id}=JSON.parse(await this.api.execute('cuda.kernel',JSON.stringify({source,entry,$session:this.session})));
        return new SharedKernel(this,id,params,block,sharedMemoryBytes,artifact,native?.defaults||{});
      },{fatal:false});
      this.kernels.set(key,pending);pending.catch(()=>this.kernels.delete(key));
    }
    return this.kernels.get(key);
  }
  batch(){this.assertAlive();return new SharedBatch(this);}
  createAccelerationStructure(options){return createAccelerationStructure(this,options);}
  rayTracingPipeline(source,options){return createRayTracingPipeline(this,source,options);}
  release(resource) {
    if(!this.resources.delete(resource))return;
    this.stats.sharedBytes-=resource.size;
    // Calls made after submit are serialized behind its GPU ownership handoff.
    const native=resource.nativeResource;
    this.tail=this.tail.then(()=>native.destroy());
    this.tail.catch(error=>{this.failure=error;});
  }
  async idle(){await this.enqueue(()=>this.api.execute('cuda.idle',JSON.stringify({$session:this.session})));}
  dispose() {
    if(this.closed)return this.done;this.closed=true;
    this.done=this.tail.catch(()=>{}).then(()=>{
      for(const resource of this.resources){resource.nativeResource.destroy();resource.destroyed=true;
        this.runtime.buffers.delete(resource);this.runtime.textures.delete(resource);}
      this.resources.clear();for(const resource of this.deviceBuffers)resource.destroyed=true;this.deviceBuffers.clear();this.deviceBytes=0;this.stats.sharedBytes=0;this.kernels.clear();this.api.close();
      for(const object of this.optixObjects)object.destroyed=true;
      this.optixObjects.clear();
    }).finally(()=>{if(owner===this)owner=null;});
    this.done.catch(()=>{});return this.done;
  }
}
class SharedKernel {
  constructor(interop,id,parameters,block,sharedMemoryBytes,artifact,defaults){Object.assign(this,{interop,id,parameters,block,sharedMemoryBytes,artifact,defaults});}
  bind(resources,scalars={}) {
    for(const key of [...Object.keys(resources),...Object.keys(scalars)])if(!this.parameters.some(p=>p.name===key))throw new TypeError('Unknown kernel parameter '+key);
    const invocation={kernel:this,resources:{...resources},scalars:{...this.defaults,...scalars},
      setScalars(values){Object.assign(this.scalars,values);return this;}};
    return invocation;
  }
}
class SharedBatch {
  constructor(interop){this.interop=interop;this.jobs=[];this.resources=new Set();this.ended=false;}
  buildAccelerationStructure(scene,vertices,options){return buildAccelerationStructure(this,scene,vertices,options);}
  trace(invocation,dimensions){return traceRays(this,invocation,dimensions);}
  dispatch(invocation,workgroups) {
    if(this.ended)throw new Error('Batch is closed.');
    const kernel=invocation.kernel;if(kernel.interop!==this.interop)throw new TypeError('Kernel belongs to another session.');
    const grid=dimensions(workgroups);
    if((!this.interop.capabilities.maxGridDimensions&&Number.isFinite(this.interop.capabilities.maxBlocksPerLaunch)&&grid.reduce((a,b)=>a*b,1)>this.interop.capabilities.maxBlocksPerLaunch)||this.jobs.length>=256)throw new RangeError('CUDA launch budget exceeded.');
    if(this.interop.capabilities.maxGridDimensions?.some((limit,axis)=>grid[axis]>limit))throw new RangeError('Grid exceeds device dimension.');
    const args=kernel.parameters.map(p=>{
      if(p.type==='buffer'||p.type==='surface') {
        const resource=invocation.resources[p.name];this.interop.check(resource,p.type);this.resources.add(resource);
        return resource.nativeBufferId?{nativeBuffer:resource.nativeBufferId}:{[p.type]:resource.nativeResource.id};
      }
      const value=invocation.scalars[p.name];if(!scalarTypes[p.type](value))throw new RangeError('Invalid scalar '+p.name);
      return {type:p.type,value};
    });
    this.jobs.push({kernel:kernel.id,grid,block:kernel.block,sharedMemoryBytes:kernel.sharedMemoryBytes,arguments:args});return this;
  }
  submit() {
    if(this.ended)throw new Error('Batch is closed.');this.ended=true;
    const resources=[...this.resources];for(const r of resources)this.interop.check(r,(r.gpuTexture||r.nativeCanvas)?'surface':'buffer');
    if(!this.jobs.length)return Promise.resolve();
    const jobs=this.jobs;
    return this.interop.enqueue(async()=>{
      for(const r of resources)this.interop.check(r,(r.gpuTexture||r.nativeCanvas)?'surface':'buffer');
      const shared=resources.filter(r=>r.nativeResource);
      const payload=JSON.stringify({jobs,$session:this.interop.session});
      const result=JSON.parse(await (shared.length?this.interop.api.dispatchShared(shared.map(r=>r.nativeResource),payload):this.interop.api.execute('cuda.dispatch',payload)));
      this.interop.stats.submissions++;this.interop.stats.dispatches+=jobs.length;return result;
    });
  }
  discard(){this.ended=true;this.jobs=[];this.resources.clear();}
}

// CUDA writes the same allocation consumed by Chromium's compositor.
// Reuse requires both CUDA completion and compositor release; submission alone
// never makes a busy surface available again.
// No canvas texture copy and no queue.onSubmittedWorkDone() in this path.
class NativeCanvasTarget {
  constructor(interop,canvas,context,surfaces) {
    Object.assign(this,{interop,canvas,context,surfaces});this.acquired=new Set();this.closed=false;
    this.width=canvas.width;this.height=canvas.height;this.cursor=0;
  }
  acquire() {
    this.interop.assertAlive();
    if(this.closed||this.surfaces.some(s=>!s.nativeResource.id))throw new DOMException('Canvas target is destroyed or lost.','InvalidStateError');
    if(this.canvas.width!==this.width||this.canvas.height!==this.height)throw new DOMException('Recreate the native target after resizing its canvas.','InvalidStateError');
    for(let i=0;i<this.surfaces.length;i++) {
      const resource=this.surfaces[this.cursor++%this.surfaces.length];
      if(!this.acquired.has(resource)&&resource.nativeResource.available) {
        this.acquired.add(resource);return resource;
      }
    }
    // Bounded backpressure: retry next animation frame. Never queue unlimited
    // native frames or overwrite a surface still being displayed.
    return null;
  }
  present(resource) {
    this.interop.assertAlive();
    if(this.closed||!this.acquired.has(resource))throw new TypeError('Present requires a frame acquired from this target.');
    resource.nativeResource.present(this.context);this.acquired.delete(resource);
  }
  cancel(resource) {this.acquired.delete(resource);}
  destroy() {
    if(this.closed)return;this.closed=true;
    for(const resource of this.surfaces){resource.destroyed=true;this.interop.release(resource);}
    this.acquired.clear();
  }
}
