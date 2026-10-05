import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GpuRuntime } from './vendor/cuda-webshader/src/runtime/runtime.js';
import { CudaSplatPipeline } from './splat-pipeline.js';
import { OceanAtmosphere } from './simulation.js';
import { CudaWaterRenderer } from './water-renderer.js';

const $ = id => document.getElementById(id);
const state = { renderer: 'pixels', swell: .8, wind: .8, ripples: .8, depth: 3, causticStrength: 1.4, reflections: .9, rays: 1, warmth: 1, clouds: true, ocean: true, sand: true, shafts: true, far: true, paused: false };
const api = window.gaussianDemo = { state, ready: false, frames: 0, time: 0, error: null };
for (const key of ['swell', 'wind', 'ripples', 'depth', 'causticStrength', 'reflections', 'rays']) $(key).addEventListener('input', () => { state[key] = Number($(key).value); $(`${key}-value`).value = state[key].toFixed(2); });
for (const key of ['clouds', 'ocean', 'sand', 'shafts','far']) $(key).addEventListener('change', () => state[key] = $(key).checked);
$('pause').onclick = () => { state.paused = !state.paused; $('pause').textContent = state.paused ? 'Resume' : 'Pause'; };
$('collapse').onclick = () => { const closed = !$('settings').hidden; $('settings').hidden = closed; $('controls').classList.toggle('closed', closed); $('collapse').textContent = closed ? '+' : '−'; $('collapse').setAttribute('aria-expanded', String(!closed)); };
for (const button of document.querySelectorAll('[data-preset]')) button.onclick = () => {
  const preset = { golden: [.8,.8,.9,1,1], blue: [.45,.35,.8,.35,0], rough: [1.8,1.6,1,1.8,.6], shallow:[.28,.6,.7,.5,.8] }[button.dataset.preset];
  ['swell','wind','reflections','rays','warmth'].forEach((key,i) => { state[key] = preset[i]; if ($(key)) { $(key).value = preset[i]; $(`${key}-value`).value = preset[i].toFixed(2); } });
  document.querySelectorAll('[data-preset]').forEach(b => b.classList.toggle('selected', b === button));
  const shallow=button.dataset.preset==='shallow';
  Object.assign(state,{depth:shallow?1.8:3,ripples:shallow?1.6:.8,causticStrength:shallow?2.4:1.4});
  for(const key of ['depth','ripples','causticStrength']){$(key).value=state[key];$(`${key}-value`).value=state[key].toFixed(2);}
  if(shallow){api.camera?.position.set(12,29,44);api.controls?.target.set(0,0,0);api.controls?.update();}
  else $('reset').click();
};
function fail(error) { api.error = String(error.stack || error); $('status').textContent = 'Pipeline stopped'; $('loading').hidden = false; $('loading-text').textContent = error.message || String(error); document.querySelector('.loader').style.display = 'none'; api.renderer?.setAnimationLoop(null); console.error(error); }
try {
  const runtime = api.runtime = await GpuRuntime.create({ backend: 'webgpu', onError: fail });
  const renderer = api.renderer = new THREE.WebGPURenderer({ device: runtime.device, antialias: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, innerWidth < 700 ? 1 : 1.5));
  renderer.setSize(innerWidth, innerHeight); renderer.setClearColor(0x23394f);
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.12;
  $('viewport').append(renderer.domElement); await renderer.init();
  const scene = new THREE.Scene(), camera = api.camera = new THREE.PerspectiveCamera(49, innerWidth / innerHeight, .25, 6000);
  const controls = api.controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.minDistance = 18; controls.maxDistance = 95;
  controls.maxPolarAngle = Math.PI * .49; controls.minPolarAngle = Math.PI * .25; controls.enablePan = false;
  const reset = () => { camera.position.set(4, 14, 72); controls.target.set(0, 10, 0); controls.update(); };
  reset(); $('reset').onclick = reset;
  const mobile = innerWidth < 700;
  const cx=mobile?48:128,cy=mobile?24:40,cz=cx,clouds=cx*cy*cz;
  const rayCount=mobile?1536:3072;
  const [sceneSource,volumeSource,fftSource,causticSource,waterSource,splatSource]=await Promise.all(['scene.cu','volume.cu','ocean-fft.cu','caustics.cu','water.cu','surface-splats.cu'].map(async file=>{const r=await fetch(file);if(!r.ok)throw Error(`Cannot load ${file}`);return r.text();}));
  const source=volumeSource+'\n'+sceneSource+'\n'+causticSource+'\n'+waterSource+'\n'+splatSource;
  const simulation=api.simulation=await OceanAtmosphere.create(runtime,{fftSource,volumeSource,sceneSource:source,n:mobile?128:256});
  const layers = [
    { name: 'clouds', entry: 'cloud_splats', count: clouds, scalars: ({time,camera:c}) => ({count:clouds,cx,cy,cz,time,wind:state.wind,warmth:state.warmth,cameraX:c.x,cameraY:c.y,cameraZ:c.z,enabled:+state.clouds}) },
    { name: 'atmosphere', entry: 'atmosphere_splats', count: 2048, scalars: () => ({count:2048,warmth:state.warmth}) },
    { name: 'rays', entry: 'ray_splats', count: rayCount, scalars: ({time}) => ({count:rayCount,time,strength:state.rays,enabled:+state.shafts,cloudEnabled:+state.clouds}) },
    { name: 'sun', entry: 'sun_splats', count: 16, scalars: () => ({count:16}) }
  ];
  const pixelPipeline = await CudaSplatPipeline.create({ renderer, runtime, source, layers, simulation });
  let pipeline=api.pipeline=pixelPipeline, splatPipeline=null, switchPending=null;
  scene.add(pixelPipeline.mesh);
  const water=api.water=await CudaWaterRenderer.create(renderer,runtime,simulation,source);scene.add(water.mesh);
  const updateModeLabel=()=>{
    $('renderer-mode').value=state.renderer;
    $('renderer-note').textContent=state.renderer==='splats'?'Ocean, reflections, sand and caustics rendered with Gaussian splats. Softer surface detail.':'Per-pixel water with Gaussian clouds.';
    $('status').textContent=`FFT ${simulation.n}² · ${state.renderer==='splats'?'All Gaussian splats':'Per-pixel water + Gaussian sky'}`;
    $('stats').textContent=`${pipeline.count.toLocaleString()} splats`;
  };
  api.setRenderer=async mode=>{
    if(!['pixels','splats'].includes(mode))throw Error('Unknown renderer mode');
    if(switchPending)await switchPending;
    if(mode==='splats'&&!splatPipeline){
      $('renderer-mode').disabled=true;$('renderer-note').textContent='Preparing Gaussian water…';
      const nx=mobile?192:512,nz=mobile?160:384,farNx=mobile?256:1024,farNz=mobile?64:128;
      const surface=(name,seabed,farfield,nx,nz)=>({name,entry:'surface_splats',count:nx*nz,
        scalars:({time,camera:c})=>({nx,nz,farfield,seabed,n:simulation.n,length:simulation.length,time,swell:state.swell,ripples:state.ripples,warmth:state.warmth,reflections:state.reflections,cameraX:c.x,cameraY:c.y,cameraZ:c.z,depth:state.depth,causticStrength:state.causticStrength,resolution:simulation.resolution,enabled:+((seabed?state.sand:state.ocean)&&(!farfield||state.far)),cloudEnabled:+state.clouds,sandEnabled:+state.sand})});
      const fullLayers=[...layers.map(layer=>({...layer})),surface('ocean',0,0,nx,nz),surface('far-ocean',0,1,farNx,farNz),surface('sand',1,0,nx,nz),surface('far-sand',1,1,farNx,farNz)];
      switchPending=CudaSplatPipeline.create({renderer,runtime,source,layers:fullLayers,simulation});
      try{splatPipeline=await switchPending;splatPipeline.mesh.visible=false;scene.add(splatPipeline.mesh);}
      finally{switchPending=null;$('renderer-mode').disabled=false;}
    }
    pipeline=api.pipeline=mode==='splats'?splatPipeline:pixelPipeline;
    pixelPipeline.mesh.visible=mode==='pixels';if(splatPipeline)splatPipeline.mesh.visible=mode==='splats';
    water.mesh.visible=mode==='pixels';state.renderer=mode;updateModeLabel();
    const url=new URL(location.href);if(mode==='splats')url.searchParams.set('renderer','splats');else url.searchParams.delete('renderer');history.replaceState(null,'',url);
  };
  $('renderer-mode').addEventListener('change',()=>api.setRenderer($('renderer-mode').value).catch(fail));
  api.renderAt = time => { api.time = time; controls.update(); pipeline.update({time,wind:state.wind,swell:state.swell,ripples:state.ripples,depth:state.depth,camera:camera.position}, camera); if(state.renderer==='pixels')water.update(time,camera,state);renderer.render(scene,camera); api.frames++; };
  if(new URL(location.href).searchParams.get('renderer')==='splats')await api.setRenderer('splats');
  api.renderAt(0); await runtime.idle();
  $('loading').hidden = true;updateModeLabel();api.ready = true;
  let last = performance.now(), sample = last, sampleFrames = 0;
  renderer.setAnimationLoop(now => {
    try {
      const dt = Math.min(.25, (now-last)/1000); last = now;
      api.renderAt(api.time + (state.paused ? 0 : dt)); sampleFrames++;
      if (now-sample > 650) { api.fps = sampleFrames*1000/(now-sample); $('stats').textContent = `${pipeline.count.toLocaleString()} splats · ${api.fps.toFixed(0)} fps`; sample=now;sampleFrames=0; }
    } catch (error) { fail(error); }
  });
  addEventListener('resize', () => { camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight); });
  if (mobile) $('collapse').click();
  api.dispose = async () => { renderer.setAnimationLoop(null); if(switchPending)await switchPending; await runtime.idle(); controls.dispose();water.dispose();pixelPipeline.dispose();splatPipeline?.dispose();simulation.dispose();renderer.dispose();runtime.dispose();api.ready=false; };
} catch (error) { fail(error); }
