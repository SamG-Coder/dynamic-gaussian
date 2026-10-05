// A dual artifact retains original CUDA, never tries to turn WGSL back into it.
// Opt in because embedding source increases every serialized kernel's size.
export function nativeArtifact(source,options,compiled){
  const m=compiled.metadata;
  if(m.textures?.length||m.surfaces?.length||m.objectHeap||m.deviceLaunchQueue||m.bufferAliases||m.libraries?.length||m.scalarConstraints?.length)
    throw new TypeError('includeNativeSource supports direct buffer/scalar kernels, not compiler-managed resources or lowered launch constraints.');
  if(new TextEncoder().encode(source).byteLength>262144)throw new RangeError('Native CUDA source exceeds 256 KiB.');
  const parameters=compiled.kernel.params.map(p=>{
    const compatiblePointer=p.pointer&&/^vec[24]<(f32|i32|u32)>$/.test(p.type);
    if(p.reference||!(['f32','i32','u32'].includes(p.type)||compatiblePointer))throw new TypeError('includeNativeSource requires float/int/uint scalars or pointers with matching native and WebGPU layout.');
    return {name:p.name,type:p.pointer?'buffer':p.type};
  });
  if(m.bindings.length!==parameters.filter(p=>p.type==='buffer').length||m.scalars.length!==parameters.filter(p=>p.type!=='buffer').length)
    throw new TypeError('includeNativeSource cannot represent synthesized bindings or constants.');
  const defaults=Object.fromEntries(m.scalars.filter(p=>p.defaultValue!==undefined).map(p=>[p.name,p.defaultValue]));
  return {version:1,source,entry:options.entry||compiled.name,workgroupSize:[...m.workgroupSize],parameters,defaults,sharedMemoryBytes:m.dynamicSharedMemoryBytes||0};
}
