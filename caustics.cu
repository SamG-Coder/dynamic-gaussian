// Sunlight refracts at the FFT surface and deposits flux on a submerged sand bed.
// Atomic fixed-point photon accumulation is read by both water and sand shaders.
__device__ float bottomDepth(float x,float z,float depth){
 float offshore=fmaxf(0.0f,-z-12.0f)*0.045f;
 float dune=0.24f*sinf(x*0.085f+z*0.065f)+0.12f*sinf(x*0.17f-z*0.055f);
 return fmaxf(0.35f,depth+offshore+dune);
}
__device__ float causticSample(const uint* caustics,float x,float z,uint resolution,float length){
 float u=(x/length-floorf(x/length))*float(resolution)-0.5f,v=(z/length-floorf(z/length))*float(resolution)-0.5f;
 uint ix=uint((int(floorf(u))+int(resolution))%int(resolution)),iz=uint((int(floorf(v))+int(resolution))%int(resolution));float fx=u-floorf(u),fz=v-floorf(v),sum=0.0f;
 for(uint dz=0u;dz<2u;dz++)for(uint dx=0u;dx<2u;dx++){
  float w=(dx==0u?1.0f-fx:fx)*(dz==0u?1.0f-fz:fz);
  sum+=float(caustics[((iz+dz)%resolution)*resolution+(ix+dx)%resolution])*w/4096.0f;
 }return sum;
}
__device__ float3 sandRadiance(const uint* caustics,float x,float z,float depth,float strength,uint resolution,float length,float warmth,float originX,float originZ){
 float ridges=sinf(x*2.0f+z*0.65f+2.0f*noise3(x*0.12f,0.0f,z*0.12f));
 float grain=noise3(x*2.2f,0.0f,z*2.2f);
 float pattern=0.91f+0.025f*ridges+0.045f*grain;
 float focused=fminf(5.0f,causticSample(caustics,x-originX+length*0.5f,z-originZ+length*0.5f,resolution,length));
 // Preserve diffuse skylight in defocused regions; strength adds focused sunlight.
 float illumination=0.68f+0.17f*fminf(1.0f,focused)+strength*0.52f*fmaxf(0.0f,focused-1.0f);
 float path=bottomDepth(x,z,depth)/0.677f;
 return make_float3(pattern*0.68f*illumination*expf(-0.18f*path),pattern*0.54f*illumination*expf(-0.075f*path),pattern*0.31f*illumination*expf(-0.035f*path));
}
__global__ void clear_caustics(uint* caustics,uint count){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)caustics[i]=0u;}
__global__ void photon_caustics(const float4* oceanMap,const float2* displacement,uint* caustics,
 uint n,uint resolution,float length,float time,float swell,float ripples,float depth,float cameraX,float cameraZ){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=resolution*resolution)return;
 float x=cameraX+(float(i%resolution)+0.5f)/float(resolution)*length-length*0.5f;
 float z=cameraZ+(float(i/resolution)+0.5f)/float(resolution)*length-length*0.5f;
 float4 wave=sampleOcean(oceanMap,x,z,n,length);float2 shift=sampleDisplacement(displacement,x,z,n,length);
 float3 ripple=rippleBand(x,z,time,ripples,cameraX,cameraZ);
 float y=wave.x*swell+ripple.x,dx=wave.y*swell+ripple.y,dz=wave.z*swell+ripple.z;
 x+=shift.x*swell;z+=shift.y*swell;
 float ni=rsqrtf(1.0f+dx*dx+dz*dz),nx=-dx*ni,ny=ni,nz=-dz*ni;
 // Incident light travels away from the sun into the water; Snell eta = air/water.
 float ix=0.22f,iy=-0.19f,iz=0.957f,cosine=-(ix*nx+iy*ny+iz*nz);
 if(cosine<=0.0f)return;
 float eta=0.7501875f,k=eta*cosine-sqrtf(1.0f-eta*eta*(1.0f-cosine*cosine));
 float tx=eta*ix+k*nx,ty=eta*iy+k*ny,tz=eta*iz+k*nz;
 if(ty>=-0.02f)return;
 float travel=(-bottomDepth(x,z,depth)-y)/ty;
 for(uint s=0u;s<2u;s++)travel=(-bottomDepth(x+tx*travel,z+tz*travel,depth)-y)/ty;
 if(travel<=0.0f)return;
 float bx=x+tx*travel,bz=z+tz*travel;
 float localX=(bx-cameraX)/length+0.5f,localZ=(bz-cameraZ)/length+0.5f;
 float u=(localX-floorf(localX))*float(resolution)-0.5f,v=(localZ-floorf(localZ))*float(resolution)-0.5f;
 int px=int(floorf(u)),pz=int(floorf(v));float fx=u-floorf(u),fz=v-floorf(v);
 float flux=4096.0f*fminf(4.0f,cosine/0.19f);
 for(uint oz=0u;oz<2u;oz++)for(uint ox=0u;ox<2u;ox++){
  uint xx=uint((px+int(ox)+int(resolution))%int(resolution)),zz=uint((pz+int(oz)+int(resolution))%int(resolution));
  float w=(ox==0u?1.0f-fx:fx)*(oz==0u?1.0f-fz:fz);
  atomicAdd(&caustics[zz*resolution+xx],uint(flux*w));
 }
}
