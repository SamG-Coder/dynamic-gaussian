// Capability probing never grants execution. The browser owns the permission
// prompt and enforces secure context, top-level visibility and user activation.
export async function SupportsNativeCuda() {
  try { return !!(await globalThis.navigator?.cuda?.SupportsNativeCuda?.()); }
  catch { return false; }
}
export async function requestPermission() {
  const api=globalThis.navigator?.cuda;
  if(typeof api?.requestPermission!=='function')throw new DOMException('This browser has no native CUDA API.','NotSupportedError');
  return api.requestPermission();
}
export async function selectNativeBackend(options={}) {
  const backend=options.backend??'auto';
  if(!['auto','native','webgpu'].includes(backend))throw new TypeError('backend must be auto, native or webgpu.');
  if(options.requestPermission!==undefined&&typeof options.requestPermission!=='boolean')throw new TypeError('requestPermission must be a boolean.');
  if(backend==='native'&&(options.device||options.adapter))throw new TypeError('A WebGPU device/adapter cannot be used with backend: native.');
  const status={available:false,permission:'unavailable',reason:null};
  if(backend==='webgpu'||options.device||options.adapter){status.available=null;status.permission='not-queried';status.reason='webgpu-requested';return {status};}
  const api=globalThis.navigator?.cuda;
  try {
    status.available=await SupportsNativeCuda();
    if(!status.available)throw new DOMException('Native CUDA is unavailable in this browser or on this device.','NotSupportedError');
    status.permission=await api.queryPermission();
    if(status.permission==='prompt'&&options.requestPermission!==false&&globalThis.navigator?.userActivation?.isActive)
      status.permission=await requestPermission();
    if(status.permission!=='granted')throw new DOMException(status.permission==='denied'?'Native GPU permission was denied.':'Native GPU permission is required. Call GpuRuntime.requestPermission() or GpuRuntime.create() from a user click.','NotAllowedError');
    const {NativeRuntime}=await import('./native-runtime.js');
    const runtime=await NativeRuntime.open(api,{...options,nativeStatus:status});
    return {runtime,status};
  } catch(error) {
    if(backend==='native')throw error;
    status.reason=!status.available?'unavailable':status.permission==='denied'?'permission-denied':status.permission!=='granted'?'permission-required':'native-initialization-failed';
    status.message=String(error.message||error);
    return {status};
  }
}
