// OptiX programs execute in the browser's existing CUDA/WebGPU ownership scope.
const scalarTypes={f32:v=>typeof v==='number'&&Number.isFinite(v)&&Number.isFinite(Math.fround(v)),
  i32:v=>Number.isInteger(v)&&v>=-2147483648&&v<=2147483647,u32:v=>Number.isInteger(v)&&v>=0&&v<=4294967295};
const elements=new Set(['void','float','float2','float3','float4','int','int2','int3','int4',
  'uint','uint2','uint3','uint4','uchar','uchar4']);
const identifier=value=>typeof value==='string'&&/^[A-Za-z_]\w{0,126}$/.test(value);
function available(interop) {
  interop.assertAlive();
  if(!interop.capabilities.optix?.available)throw new Error(interop.capabilities.optix?.reason||'OptiX is unavailable.');
}
function check(object,interop,kind) {
  interop.assertAlive();
  if(object?.interop!==interop||object.kind!==kind||object.destroyed||!interop.optixObjects.has(object))
    throw new TypeError('OptiX object is destroyed or belongs to another session.');
}
class OptixObject {
  constructor(interop,id,kind){Object.assign(this,{interop,id,kind,destroyed:false});interop.optixObjects.add(this);}
  destroy() {
    if(this.destroyed)return this.done||Promise.resolve();
    this.destroyed=true;this.interop.optixObjects.delete(this);
    if(this.interop.closed)return Promise.resolve();
    return this.done=this.interop.enqueue(()=>this.interop.api.execute(
      this.kind==='scene'?'cuda.optix.destroyScene':'cuda.optix.destroyPipeline',
      JSON.stringify({id:this.id,$session:this.interop.session})),{fatal:false});
  }
}
export async function createAccelerationStructure(interop,{vertexCount,vertexStride=12,allowUpdate=true}={}) {
  available(interop);
  if(!Number.isInteger(vertexCount)||vertexCount<3||vertexCount%3||vertexCount>interop.capabilities.optix.maxVertices||
    !Number.isInteger(vertexStride)||vertexStride<12||vertexStride>256||vertexStride%4||typeof allowUpdate!=='boolean')
    throw new RangeError('OptiX geometry requires non-indexed float3 triangles and a 4-byte-aligned stride of 12..256 bytes.');
  return interop.enqueue(async()=>{
    const {id}=JSON.parse(await interop.api.execute('cuda.optix.scene',JSON.stringify({vertexCount,vertexStride,allowUpdate,$session:interop.session})));
    return Object.assign(new OptixObject(interop,id,'scene'),{vertexCount,vertexStride,allowUpdate});
  },{fatal:false});
}
export async function createRayTracingPipeline(interop,source,{raygen='__raygen__main',miss='__miss__main',
  closestHit='__closesthit__main',anyHit='',parameters=[],maxTraceDepth=2,numPayloadValues=8}={}) {
  available(interop);
  if(typeof source!=='string'||!source.length||new TextEncoder().encode(source).length>262144)
    throw new RangeError('OptiX CUDA source must fit 256 KiB.');
  for(const [name,prefix,optional] of [[raygen,'__raygen__'],[miss,'__miss__'],[closestHit,'__closesthit__'],[anyHit,'__anyhit__',true]])
    if(!(optional&&name==='')&&(!identifier(name)||!name.startsWith(prefix)))throw new TypeError('Invalid OptiX entry name.');
  if(!Number.isInteger(maxTraceDepth)||maxTraceDepth<1||maxTraceDepth>interop.capabilities.optix.maxTraceDepth||
    !Number.isInteger(numPayloadValues)||numPayloadValues<0||numPayloadValues>interop.capabilities.optix.maxPayloadValues)
    throw new RangeError('Unsupported OptiX trace depth or payload count.');
  if(!Array.isArray(parameters)||parameters.length>32||parameters.some(p=>!p||!identifier(p.name)||p.name==='scene'||
    !['buffer','surface','f32','i32','u32'].includes(p.type)||(p.element!==undefined&&(p.type!=='buffer'||!elements.has(p.element))))||
    new Set(parameters.map(p=>p.name)).size!==parameters.length)throw new TypeError('Invalid OptiX parameter descriptors.');
  const params=parameters.map(p=>({...p}));
  return interop.enqueue(async()=>{
    const {id}=JSON.parse(await interop.api.execute('cuda.optix.pipeline',JSON.stringify({source,raygen,miss,closestHit,anyHit,
      parameters:params,maxTraceDepth,numPayloadValues,$session:interop.session})));
    return new OptixPipeline(interop,id,params);
  },{fatal:false});
}
class OptixPipeline extends OptixObject {
  constructor(interop,id,parameters){super(interop,id,'pipeline');this.parameters=parameters;}
  bind(scene,resources={},scalars={}) {
    check(this,this.interop,'pipeline');check(scene,this.interop,'scene');
    for(const key of [...Object.keys(resources),...Object.keys(scalars)])
      if(!this.parameters.some(p=>p.name===key))throw new TypeError('Unknown OptiX parameter '+key);
    return {pipeline:this,scene,resources:{...resources},scalars:{...scalars},
      setScalars(values){Object.assign(this.scalars,values);return this;}};
  }
}
export function buildAccelerationStructure(batch,scene,vertices,{update=false,vertexOffset=0}={}) {
  const interop=batch.interop;check(scene,interop,'scene');interop.check(vertices,'buffer');if(!vertices.nativeResource)throw new TypeError('Acceleration structure construction requires shared geometry.');
  if(batch.ended)throw new Error('Batch is closed.');
  if(batch.jobs.length>=256)throw new RangeError('GPU job budget exceeded.');
  if(typeof update!=='boolean'||(update&&!scene.allowUpdate)||!Number.isInteger(vertexOffset)||vertexOffset<0||vertexOffset%4||
    vertexOffset+(scene.vertexCount-1)*scene.vertexStride+12>vertices.size)throw new RangeError('Invalid OptiX geometry range or update.');
  batch.resources.add(vertices);
  batch.jobs.push({type:'optix-build',scene:scene.id,vertexOffset,update,arguments:[{buffer:vertices.nativeResource.id}]});
  return batch;
}
export function traceRays(batch,invocation,dimensions) {
  const {interop}=batch,{pipeline,scene}=invocation;check(pipeline,interop,'pipeline');check(scene,interop,'scene');
  if(batch.ended)throw new Error('Batch is closed.');
  const extent=Array.isArray(dimensions)?[...dimensions]:[dimensions];while(extent.length<3)extent.push(1);
  if(batch.jobs.length>=256||extent.length!==3||extent[2]!==1||extent.some(n=>!Number.isInteger(n)||n<1||n>8192)||
    extent[0]*extent[1]>interop.capabilities.optix.maxLaunchPixels)throw new RangeError('OptiX launch exceeds supported dimensions.');
  const args=pipeline.parameters.map(p=>{
    if(p.type==='buffer'||p.type==='surface') {
      const resource=invocation.resources[p.name];interop.check(resource,p.type);batch.resources.add(resource);
      return resource.nativeBufferId?{nativeBuffer:resource.nativeBufferId}:{[p.type]:resource.nativeResource.id};
    }
    const value=invocation.scalars[p.name];if(!scalarTypes[p.type](value))throw new RangeError('Invalid OptiX scalar '+p.name);
    return {type:p.type,value};
  });
  // Even a shader without buffer/surface parameters needs ownership of a
  // browser-managed resource. Typical renderers bind their output here.
  if(![...batch.resources].some(r=>r.nativeResource))
    throw new TypeError('OptiX launches require at least one shared resource in the batch.');
  batch.jobs.push({type:'optix-trace',pipeline:pipeline.id,scene:scene.id,dimensions:extent,arguments:args});
  return batch;
}
