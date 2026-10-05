/** Precompiled, allocation-free frame schedule for spectral water and cloud volumes. */
export class OceanAtmosphere {
  static async create(runtime, {fftSource,volumeSource,sceneSource,n=256}) {
    if (![128,256].includes(n)) throw new RangeError('FFT resolution must be 128 or 256');
    const s=new OceanAtmosphere();Object.assign(s,{runtime,n,length:256,resolution:n*4,volumeCount:96*40*96,invocations:{},resources:{}});
    const sizes={spectrum:n*n*3*8,scratch:n*n*3*8,spatial:n*n*3*8,oceanMap:n*n*16,displacement:n*n*8,density:s.volumeCount*16,volume:s.volumeCount*16,caustics:s.resolution*s.resolution*4};
    for(const [name,bytes] of Object.entries(sizes))s.resources[name]=runtime.createBuffer(bytes,{label:name});
    const r=s.resources;
    const make=async(entry,source,bindings,values,block=128)=>{
      const k=await runtime.kernel(source,{entry,workgroupSize:[block,1,1]});return k.bind(bindings,values);
    };
    try {
      s.invocations.evolve=await make('evolve_spectrum',fftSource,{spectrum:r.spectrum},{n,length:s.length,time:0,windSpeed:18});
      s.invocations.rows=await make('inverse_fft',fftSource,{input:r.spectrum,output:r.scratch},{n,axis:0});
      s.invocations.columns=await make('inverse_fft',fftSource,{input:r.scratch,output:r.spatial},{n,axis:1});
      s.invocations.surface=await make('ocean_surface',fftSource,{spatial:r.spatial,oceanMap:r.oceanMap,displacement:r.displacement},{n,length:s.length});
      s.invocations.density=await make('cloud_density',volumeSource,{density:r.density},{count:s.volumeCount,time:0,wind:.8});
      s.invocations.lighting=await make('cloud_lighting',volumeSource,{density:r.density,volume:r.volume},{count:s.volumeCount});
      s.invocations.clear=await make('clear_caustics',sceneSource,{caustics:r.caustics},{count:s.resolution*s.resolution});
      s.invocations.photons=await make('photon_caustics',sceneSource,{oceanMap:r.oceanMap,displacement:r.displacement,caustics:r.caustics},{n,resolution:s.resolution,length:s.length,time:0,swell:.8,ripples:.8,depth:3,cameraX:4,cameraZ:72});
    } catch(error){s.dispose();throw error;}
    return s;
  }
  record(batch,{time,wind,swell,ripples,depth,camera}) {
    const k=this.invocations,n=this.n;
    batch.dispatch(k.evolve.setScalars({time,windSpeed:14+wind*5}),[Math.ceil(n*n/128)]);
    batch.dispatch(k.rows,[n*3]);batch.dispatch(k.columns,[n*3]);
    batch.dispatch(k.surface,[Math.ceil(n*n/128)]);
    const volumeGroups=Math.ceil(this.volumeCount/128);
    batch.dispatch(k.density.setScalars({time,wind}),[volumeGroups]);batch.dispatch(k.lighting,[volumeGroups]);
    const photonGroups=Math.ceil(this.resolution*this.resolution/128);
    batch.dispatch(k.clear,[photonGroups]);batch.dispatch(k.photons.setScalars({time,swell,ripples,depth,cameraX:camera.x,cameraZ:camera.z}),[photonGroups]);
  }
  dispose(){for(const r of Object.values(this.resources))if(!r.destroyed)this.runtime.destroyBuffer(r);}
}
