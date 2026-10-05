// Shared density/light field used by visible cloud splats, ocean rays, and shafts.
// Full surrounding world box: [-240, 8, -240] .. [240, 160, 240].
__device__ float vhash(uint n){n^=n>>16;n*=2146121005u;n^=n>>15;n*=2221713035u;n^=n>>16;return float(n&16777215u)/16777215.0f;}
__device__ float vlerp(float a,float b,float t){return a+(b-a)*t;}
__device__ float noise3(float x,float y,float z){
 int ix=int(floorf(x)),iy=int(floorf(y)),iz=int(floorf(z));
 float u=x-floorf(x),v=y-floorf(y),w=z-floorf(z);u=u*u*(3.0f-2.0f*u);v=v*v*(3.0f-2.0f*v);w=w*w*(3.0f-2.0f*w);
 uint h=uint(ix)*73856093u+uint(iy)*19349663u+uint(iz)*83492791u;
 float a=vlerp(vhash(h),vhash(h+73856093u),u),b=vlerp(vhash(h+19349663u),vhash(h+19349663u+73856093u),u);
 float c=vlerp(vhash(h+83492791u),vhash(h+83492791u+73856093u),u),d=vlerp(vhash(h+83492791u+19349663u),vhash(h+83492791u+19349663u+73856093u),u);
 return vlerp(vlerp(a,b,v),vlerp(c,d,v),w);
}
// Wind advection and buoyant domain warping evolve one continuous surrounding field.
// Weather coverage, billows and fine erosion have independent spatial/time scales.
__device__ float densityAt(float x,float y,float z,float time,float wind){
 float edge=fminf(1.0f,fmaxf(0.0f,(240.0f-fmaxf(fabsf(x),fabsf(z)))/24.0f));
 x-=time*wind*4.5f;z-=time*wind*1.4f;
 float growth=time*0.075f;
 float warp=noise3(x*0.035f,growth,z*0.035f)-0.5f;
 float weather=noise3(x*0.029f+warp*0.7f,2.7f+growth*0.32f,z*0.029f);
 float base=52.0f+7.0f*noise3(x*0.018f,6.0f,z*0.018f);
 float thickness=16.0f+42.0f*noise3(x*0.022f,9.3f+growth*0.23f,z*0.022f);
 float h=(y-base)/thickness;
 if(h<=0.0f||h>=1.0f)return 0.0f;
 float low=fminf(1.0f,h*9.0f),high=fminf(1.0f,(1.0f-h)*2.3f);
 low=low*low*(3.0f-2.0f*low);high=high*high*(3.0f-2.0f*high);
 float profile=low*high;
 float billow=noise3(x*0.095f+warp,y*0.095f-growth,z*0.095f);
 float erosion=noise3(x*0.23f,y*0.23f-growth*1.6f,z*0.23f)*0.65f+noise3(x*0.47f,y*0.47f-growth*2.1f,z*0.47f)*0.35f;
 float shape=fminf(1.0f,fmaxf(0.0f,((weather-0.60f)*1.7f+(billow-0.42f)*0.75f-(1.0f-profile)*0.4f-(1.0f-erosion)*0.1f)*4.0f));
 return shape*profile*edge*0.18f;
}
// Approximate higher-order scattering lifts shadowed interiors; sun transmittance
// still controls directional rims. Shared by primary splats and reflected rays.
__device__ float3 cloudRadiance(float sunT,float skyT,float mu,float warmth){
 float phase=0.32f+0.35f*(1.0f-0.42f*0.42f)/powf(1.0f+0.42f*0.42f-0.84f*mu,1.5f);
 float sun=sunT*phase*0.65f;
 float multiple=0.22f+0.26f*sqrtf(sunT)+0.18f*sqrtf(skyT);
 return make_float3(multiple*0.94f+sun*(1.0f+0.12f*warmth),multiple+sun*0.96f,multiple*1.07f+sun*(0.94f-0.12f*warmth));
}
__device__ float4 sampleVolume(const float4* volume,float x,float y,float z){
 float gx=(x+240.0f)/480.0f*95.0f,gy=(y-8.0f)/152.0f*39.0f,gz=(z+240.0f)/480.0f*95.0f;
 if(gx<0.0f||gx>=95.0f||gy<0.0f||gy>=39.0f||gz<0.0f||gz>=95.0f)return make_float4(0.0f,1.0f,1.0f,0.0f);
 uint ix=uint(gx),iy=uint(gy),iz=uint(gz);float u=gx-float(ix),v=gy-float(iy),w=gz-float(iz);
 float4 result=make_float4(0.0f,0.0f,0.0f,0.0f);
 for(uint dz=0u;dz<2u;dz++)for(uint dy=0u;dy<2u;dy++)for(uint dx=0u;dx<2u;dx++){
  float weight=(dx==0u?1.0f-u:u)*(dy==0u?1.0f-v:v)*(dz==0u?1.0f-w:w);
  float4 a=volume[((iz+dz)*40u+iy+dy)*96u+ix+dx];
  result.x+=a.x*weight;result.y+=a.y*weight;result.z+=a.z*weight;
 }
 return result;
}
__global__ void cloud_density(float4* density,uint count,float time,float wind){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 uint x=i%96u,y=(i/96u)%40u,z=i/(96u*40u);
 float wx=-240.0f+float(x)*480.0f/95.0f,wy=8.0f+float(y)*152.0f/39.0f,wz=-240.0f+float(z)*480.0f/95.0f;
 density[i]=make_float4(densityAt(wx,wy,wz,time,wind),1.0f,1.0f,0.0f);
}
__global__ void cloud_lighting(const float4* density,float4* volume,uint count){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 uint x=i%96u,y=(i/96u)%40u,z=i/(96u*40u);
 float wx=-240.0f+float(x)*480.0f/95.0f,wy=8.0f+float(y)*152.0f/39.0f,wz=-240.0f+float(z)*480.0f/95.0f;
 float optical=0.0f,ambient=0.0f;
 for(uint step=0u;step<12u;step++){
  float t=(float(step)+0.5f)*8.0f;
  optical+=sampleVolume(density,wx-0.22f*t,wy+0.19f*t,wz-0.957f*t).x*8.0f;
  if(step<6u)ambient+=sampleVolume(density,wx,wy+t,wz).x*8.0f;
 }
 volume[i]=make_float4(density[i].x,expf(-optical),expf(-ambient*0.5f),0.0f);
}
// Front-to-back Beer-Lambert integration through the exact same baked volume.
// Returns linear radiance and remaining background transmittance.
__device__ float4 traceCloud(const float4* volume,float ox,float oy,float oz,float dx,float dy,float dz,float warmth){
 float invx=1.0f/(fabsf(dx)<0.0001f?0.0001f:dx),invy=1.0f/(fabsf(dy)<0.0001f?0.0001f:dy),invz=1.0f/(fabsf(dz)<0.0001f?0.0001f:dz);
 float ax=(-240.0f-ox)*invx,bx=(240.0f-ox)*invx,ay=(8.0f-oy)*invy,by=(160.0f-oy)*invy,az=(-240.0f-oz)*invz,bz=(240.0f-oz)*invz;
 float nearT=fmaxf(0.0f,fmaxf(fminf(ax,bx),fmaxf(fminf(ay,by),fminf(az,bz))));
 float farT=fminf(fmaxf(ax,bx),fminf(fmaxf(ay,by),fmaxf(az,bz)));
 if(farT<=nearT)return make_float4(0.0f,0.0f,0.0f,1.0f);
 float step=(farT-nearT)/40.0f,T=1.0f,r=0.0f,g=0.0f,b=0.0f;
 float mu=dx*(-0.22f)+dy*0.19f+dz*(-0.957f);
 for(uint s=0u;s<40u;s++){
  float t=nearT+(float(s)+0.5f)*step;float4 f=sampleVolume(volume,ox+dx*t,oy+dy*t,oz+dz*t);
  float alpha=1.0f-expf(-f.x*step);float3 light=cloudRadiance(f.y,f.z,mu,warmth);
  r+=T*alpha*light.x;g+=T*alpha*light.y;b+=T*alpha*light.z;
  T*=1.0f-alpha;if(T<0.015f)break;
 }
 return make_float4(r,g,b,T);
}
