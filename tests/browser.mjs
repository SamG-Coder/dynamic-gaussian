import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import { createStaticServer } from '../scripts/serve.mjs';

const evidence = new URL(`../${process.env.EVIDENCE_DIR || 'evidence'}/`, import.meta.url);
await mkdir(evidence, { recursive: true });
const server = createStaticServer(process.env.TEST_ROOT);
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = process.env.TEST_URL || `http://127.0.0.1:${server.address().port}/`;
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const errors = [];
const report = { passed: false, softwareAdapterRequested: false };
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto(url);
  await page.waitForFunction(() => window.gaussianDemo?.ready || window.gaussianDemo?.error, null, { timeout: 60000 });
  const error = await page.evaluate(() => gaussianDemo.error); if (error) throw new Error(error);
  await page.waitForFunction(() => gaussianDemo.frames > 60);
  report.gpu = await page.evaluate(async () => {
    const d = gaussianDemo, { runtime: rt, pipeline: p } = d;
    const assert = (value, message) => { if (!value) throw new Error(message); };
    d.renderer.setAnimationLoop(null);
    const readbackBefore = rt.stats.readbackBytes, uploadsBefore = rt.stats.dataBytesUploaded;
    for (let i=0; i<12; i++) d.renderAt(i*.12);
    await rt.idle();
    assert(rt.stats.readbackBytes === readbackBefore, 'Frame loop read back scene data');
    assert(rt.stats.dataBytesUploaded === uploadsBefore, 'Frame loop uploaded scene data');
    const read = name => rt.read(p.resources[name], name === 'colors' ? Uint32Array : Float32Array);
    d.renderAt(3); await rt.idle();
    const centers = await read('centers'), a = await read('covA'), b = await read('covB'), colors = await read('colors');
    assert(centers.length === p.count*4, 'Wrong buffer size');
    let minDeterminant = Infinity;
    for (let i=0;i<p.count;i++) {
      const j=i*4, xx=a[j], xy=a[j+1], xz=a[j+2], yy=a[j+3], yz=b[j], zz=b[j+1];
      assert([centers[j],centers[j+1],centers[j+2],xx,xy,xz,yy,yz,zz].every(Number.isFinite), `Nonfinite splat ${i}`);
      const det=xx*(yy*zz-yz*yz)-xy*(xy*zz-xz*yz)+xz*(xy*yz-xz*yy);
      minDeterminant=Math.min(minDeterminant,det);
      assert(xx>0 && xx*yy-xy*xy>0 && det>0, `Invalid covariance ${i}`);
    }
    // Verify one global GPU sort produces a complete permutation ordered by depth bins.
    const attr=p.mesh._sort.orderAttribute;
    const orderResource=rt.importBuffer(d.renderer.backend.get(attr).buffer, p.count*4, 'validation only');
    const order=await rt.read(orderResource,Uint32Array); rt.destroyBuffer(orderResource);
    assert(new Set(order).size===p.count && order.every(i=>i<p.count),'Sort is not a permutation');
    const matrix=p.mesh._sortMatrix.value.elements;
    let previous=-1;
    for(const i of order) {
      const j=i*4, depth=-(matrix[2]*centers[j]+matrix[6]*centers[j+1]+matrix[10]*centers[j+2]+matrix[14]);
      const bin=4095-Math.floor(Math.max(0,Math.min(1,Math.log2(Math.max(depth,.25)/.25)/Math.log2(6000/.25)))*4095);
      assert(bin+1>=previous,'GPU depth order invalid'); previous=bin;
    }
    const waterRead=()=>rt.read(d.water.resource,Uint32Array);
    const difference=(a,b)=>a.reduce((sum,v,i)=>sum+(v!==b[i]?1:0),0);
    const initialWater=await waterRead();
    assert(initialWater.some(v=>(v>>>24)===0)&&initialWater.some(v=>(v>>>24)===255),'Water/sky alpha mask missing');
    const clouds=p.layers.find(l=>l.name==='clouds'),quadrants=[0,0,0,0];
    let luminance=0,litCount=0;
    for(let i=clouds.offset;i<clouds.offset+clouds.count;i++)if((colors[i]>>>24)>8){
      quadrants[(centers[i*4]>0?1:0)+(centers[i*4+2]>0?2:0)]++;
      luminance+=((colors[i]&255)+((colors[i]>>>8)&255)+((colors[i]>>>16)&255))/(255*3);litCount++;
    }
    assert(quadrants.every(count=>count>100),'Clouds do not surround the camera');
    assert(luminance/litCount>.32,'Clouds have excessively dark lighting');
    d.renderAt(8);await rt.idle();const later=await read('centers'),laterColors=await read('colors');
    const waterAnimation=difference(initialWater,await waterRead());
    let moving=0;for(let i=0;i<clouds.count;i++)if(Math.abs(later[i*4]-centers[i*4])>1)moving++;
    assert(moving>100 && waterAnimation>1000,'Wind/water animation missing');
    const cloudShapeChanges=difference(colors,laterColors);assert(cloudShapeChanges>100,'Cloud shapes do not evolve');
    d.state.wind=0;d.renderAt(3);await rt.idle();const stillWeather=await rt.read(d.simulation.resources.density,Float32Array);
    d.renderAt(14);await rt.idle();const grownWeather=await rt.read(d.simulation.resources.density,Float32Array);
    const cloudFormation=difference(stillWeather,grownWeather);assert(cloudFormation>100,'Clouds only translate, without forming');d.state.wind=.8;
    for(const [name,key] of [['clouds','clouds'],['rays','shafts']]){
      d.state[key]=false;d.renderAt(3);await rt.idle();const hidden=await read('colors');const layer=p.layers.find(l=>l.name===name);
      assert(hidden.subarray(layer.offset,layer.offset+layer.count).every(v=>(v>>>24)===0),`${name} toggle failed`);d.state[key]=true;
    }
    d.state.reflections=0;d.renderAt(3);await rt.idle();const reflectionChanges=difference(initialWater,await waterRead());
    assert(reflectionChanges>1000,'Reflection control does not affect water pixels');d.state.reflections=.9;
    d.renderAt(3);await rt.idle();const beforeCamera=await waterRead();d.camera.position.x+=12;d.renderAt(3);await rt.idle();const viewChanges=difference(beforeCamera,await waterRead());
    assert(viewChanges>1000,'Water does not respond to camera');document.getElementById('reset').click();
    // Caustics must affect the refracted seabed, and never the water reflection alone.
    d.state.sand=false;d.state.causticStrength=0;d.renderAt(3);await rt.idle();const noSand=await waterRead();
    d.state.causticStrength=3;d.renderAt(3);await rt.idle();const causticsWithoutSand=difference(noSand,await waterRead());
    assert(causticsWithoutSand===0,'Caustics are applied to the water surface');
    d.state.sand=true;d.state.ocean=false;d.state.causticStrength=0;d.renderAt(3);await rt.idle();const unlitSand=await waterRead();
    d.state.causticStrength=3;d.renderAt(3);await rt.idle();const litSand=await waterRead();const sandCausticChanges=difference(unlitSand,litSand);
    assert(sandCausticChanges>1000,'Caustics do not illuminate seabed');
    for(let i=0;i<litSand.length;i++)for(const shift of [0,8,16])assert(((litSand[i]>>>shift)&255)>=((unlitSand[i]>>>shift)&255),'Caustic strength darkens sand');
    d.state.ocean=true;d.state.causticStrength=1.4;d.state.depth=8;d.renderAt(3);await rt.idle();const depthChanges=difference(initialWater,await waterRead());
    assert(depthChanges>1000,'Water depth does not change refraction/absorption');d.state.depth=3;
    const cardinalCoverage=[];
    for(const [x,z] of [[0,72],[72,0],[0,-72],[-72,0]]){
      d.camera.position.set(x,14,z);d.renderAt(3);await rt.idle();const pixels=await waterRead();const coverage=pixels.reduce((n,v)=>n+((v>>>24)>0?1:0),0);
      assert(coverage>pixels.length*.25,'Water absent in a cardinal direction');cardinalCoverage.push(coverage);
    }
    document.getElementById('reset').click();d.renderAt(3);await rt.idle();
    const simulation=d.simulation, sr=simulation.resources, n=simulation.n;
    const spectrum=await rt.read(sr.spectrum,Float32Array),spatial=await rt.read(sr.spatial,Float32Array),map=await rt.read(sr.oceanMap,Float32Array),volume=await rt.read(sr.volume,Float32Array);
    let hermitianError=0,maxImaginary=0,heightEnergy=0,nonempty=0,shadowed=0;
    for(let z=0;z<n;z++)for(let x=0;x<n;x++){
      const i=z*n+x,m=((n-z)%n)*n+(n-x)%n;
      hermitianError=Math.max(hermitianError,Math.abs(spectrum[i*2]-spectrum[m*2]),Math.abs(spectrum[i*2+1]+spectrum[m*2+1]));
      maxImaginary=Math.max(maxImaginary,Math.abs(spatial[i*2+1]/(n*n)));
      heightEnergy+=map[i*4]*map[i*4];
    }
    const heightRms=Math.sqrt(heightEnergy/(n*n));
    assert(hermitianError<.05 && maxImaginary<1e-4,'Spectral water is not a real Hermitian IFFT');
    assert(heightRms>.05 && heightRms<5,'FFT height energy out of range');
    for(let i=0;i<volume.length;i+=4){assert(volume[i]>=0 && volume[i+1]>=0 && volume[i+1]<=1 && Number.isFinite(volume[i+2]),'Invalid cloud volume');if(volume[i]>.01){nonempty++;if(volume[i+1]<.5)shadowed++;}}
    assert(nonempty>100 && shadowed>50,'Cloud density or light marching missing');
    const photonMap=await rt.read(sr.caustics,Uint32Array);let flux=0,flux2=0;
    for(const value of photonMap){flux+=value/4096;flux2+=(value/4096)**2;}
    const meanFlux=flux/photonMap.length,fluxDeviation=Math.sqrt(flux2/photonMap.length-meanFlux**2);
    assert(meanFlux>.1 && meanFlux<2 && fluxDeviation>.05,'Caustic flux has no focus or invalid energy');
    d.renderAt(8);await rt.idle();const causticAnimation=difference(photonMap,await rt.read(sr.caustics,Uint32Array));
    assert(causticAnimation>1000,'Caustic focusing does not animate');
    d.state.ripples=0;d.renderAt(3);await rt.idle();const noRipples=await waterRead();d.state.ripples=.8;d.renderAt(3);await rt.idle();const rippleChanges=difference(noRipples,await waterRead());
    assert(rippleChanges>1000,'Ripple control does not change water pixels');
    d.state.far=false;d.renderAt(3);await rt.idle();const withoutFar=await waterRead();
    assert(withoutFar.filter(v=>(v>>>24)>0).length<initialWater.filter(v=>(v>>>24)>0).length,'Far-water switch failed');
    d.state.far=true;d.renderAt(3);await rt.idle();
    // Independent tiny direct inverse DFT checks the same row/column GPU FFT.
    const small=8,cells=small*small,inputs=new Float32Array(cells*3*2);
    for(let i=0;i<inputs.length;i++)inputs[i]=Math.sin(i*1.17)*.2;
    const ir=rt.createBuffer(inputs),scratch=rt.createBuffer(inputs.byteLength),out=rt.createBuffer(inputs.byteLength);
    const fftKernel=simulation.invocations.rows.kernel;
    rt.batch().dispatch(fftKernel.bind({input:ir,output:scratch},{n:small,axis:0}),[small*3]).dispatch(fftKernel.bind({input:scratch,output:out},{n:small,axis:1}),[small*3]).submit();
    const actual=await rt.read(out,Float32Array);let dftError=0;
    for(let field=0;field<3;field++)for(let y=0;y<small;y++)for(let x=0;x<small;x++){
      let re=0,im=0;for(let ky=0;ky<small;ky++)for(let kx=0;kx<small;kx++){
        const k=(field*cells+ky*small+kx)*2,angle=2*Math.PI*(x*kx+y*ky)/small;
        re+=inputs[k]*Math.cos(angle)-inputs[k+1]*Math.sin(angle);im+=inputs[k]*Math.sin(angle)+inputs[k+1]*Math.cos(angle);
      }
      const k=(field*cells+y*small+x)*2;dftError=Math.max(dftError,Math.abs(re-actual[k]),Math.abs(im-actual[k+1]));
    }
    [ir,scratch,out].forEach(r=>rt.destroyBuffer(r));assert(dftError<1e-4,'GPU FFT differs from direct DFT');
    assert(p.mesh.splatGeometry.attributes.position.array.every(v=>v===0),'CPU seed positions were modified');
    return { device:rt.describe(),splats:p.count,minDeterminant,moving,waterAnimation,cloudShapeChanges,cloudFormation,quadrants,cloudMeanRadiance:luminance/litCount,reflectionChanges,viewChanges,
      globalDepthSort:'logarithmic 4096 bins',frameReadbackBytes:0,frameSceneUploadBytes:0,frames:d.frames,observedFps:d.fps,
      fft:{n,heightRms,hermitianError,maxImaginary,dftError},cloudVolume:{cells:volume.length/4,nonempty,shadowed},cardinalCoverage,rippleChanges,waterPixels:initialWater.length,caustics:{resolution:simulation.resolution,meanFlux,fluxDeviation,causticAnimation,causticsWithoutSand,sandCausticChanges,depthChanges} };
  });
  await page.screenshot({ path: new URL('desktop.png', evidence).pathname.replace(/^\/([A-Za-z]:)/, '$1') });
  await page.evaluate(async()=>{gaussianDemo.renderAt(14);await gaussianDemo.runtime.idle();});
  await page.screenshot({path:new URL('clouds-later.png',evidence).pathname.replace(/^\/([A-Za-z]:)/,'$1')});
  await page.evaluate(async()=>{gaussianDemo.camera.position.set(0,14,-72);gaussianDemo.renderAt(3);await gaussianDemo.runtime.idle();});
  await page.screenshot({path:new URL('reverse-view.png',evidence).pathname.replace(/^\/([A-Za-z]:)/,'$1')});
  await page.getByRole('button',{name:'Shallows',exact:true}).click();
  await page.evaluate(async()=>{gaussianDemo.renderAt(3);await gaussianDemo.runtime.idle();});
  await page.screenshot({path:new URL('shallows.png',evidence).pathname.replace(/^\/([A-Za-z]:)/,'$1')});
  await page.evaluate(async()=>{gaussianDemo.state.ocean=false;gaussianDemo.renderAt(3);await gaussianDemo.runtime.idle();});
  await page.screenshot({path:new URL('sand-only.png',evidence).pathname.replace(/^\/([A-Za-z]:)/,'$1')});
  await page.evaluate(()=>{gaussianDemo.state.ocean=true;document.getElementById('reset').click();});
  await page.getByRole('button', {name:'Open sea',exact:true}).click();
  await page.evaluate(async()=>{gaussianDemo.renderAt(3);await gaussianDemo.runtime.idle();});
  if (await page.locator('#swell').inputValue() !== '1.8') throw new Error('Preset UI failed');
  await page.screenshot({ path: new URL('open-sea.png', evidence).pathname.replace(/^\/([A-Za-z]:)/, '$1') });
  await page.getByRole('button',{name:'Pause',exact:true}).click();
  if (await page.getByRole('button',{name:'Resume',exact:true}).count() !== 1) throw new Error('Pause UI failed');
  await page.evaluate(()=>gaussianDemo.dispose());
  report.desktopDisposed=true;
  const mobile = await browser.newPage({viewport:{width:390,height:844},deviceScaleFactor:1,isMobile:true,hasTouch:true});
  mobile.on('pageerror',error=>errors.push(error.message));
  await mobile.goto(url);
  await mobile.waitForFunction(()=>window.gaussianDemo?.ready||window.gaussianDemo?.error,null,{timeout:60000});
  report.mobile=await mobile.evaluate(async()=>{
    if(gaussianDemo.error)throw Error(gaussianDemo.error);
    gaussianDemo.renderer.setAnimationLoop(null);gaussianDemo.renderAt(3);await gaussianDemo.runtime.idle();
    return {splats:gaussianDemo.pipeline.count,noOverflow:document.documentElement.scrollWidth===innerWidth,controlsCollapsed:document.getElementById('settings').hidden};
  });
  if(!report.mobile.noOverflow||!report.mobile.controlsCollapsed)throw Error('Mobile layout failed');
  await mobile.locator('#collapse').tap();
  await mobile.locator('#clouds').tap();
  if(await mobile.evaluate(()=>gaussianDemo.state.clouds)!==false)throw Error('Touch layer control failed');
  await mobile.locator('#clouds').tap();await mobile.locator('#collapse').tap();
  await mobile.evaluate(async()=>{gaussianDemo.renderAt(3);await gaussianDemo.runtime.idle();});
  await mobile.screenshot({path:new URL('mobile.png',evidence).pathname.replace(/^\/([A-Za-z]:)/,'$1')});
  await mobile.evaluate(()=>gaussianDemo.dispose());
  if(errors.length)throw Error(errors.join('\n'));
  report.passed=true;
} catch(error) { report.error=String(error.stack||error);process.exitCode=1; }
finally {
  await writeFile(new URL('validation.json',evidence),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report,null,2));
  await browser.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
}
