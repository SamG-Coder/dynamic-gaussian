// Compiled with volume.cu. Only CUDA writes positions, covariances, and colors.
// Covariance ABI: A=(xx,xy,xz,yy), B=(yz,zz,0,0); packed linear RGBA8.
__device__ float sat(float x){return fminf(1.0f,fmaxf(0.0f,x));}
__device__ uint rgba(float r,float g,float b,float a){return uint(sat(r)*255.0f)|(uint(sat(g)*255.0f)<<8)|(uint(sat(b)*255.0f)<<16)|(uint(sat(a)*255.0f)<<24);}
__device__ float3 skyColor(float elevation,float warmth){
 float h=sat(elevation*2.0f);
 return make_float3(0.35f*(1.0f-h)+0.026f*h+warmth*0.12f*(1.0f-h),0.32f*(1.0f-h)+0.075f*h,0.34f*(1.0f-h)+0.16f*h);
}
__device__ float4 sampleOcean(const float4* map,float x,float z,uint n,float length){
 float u=(x/length-floorf(x/length))*float(n),v=(z/length-floorf(z/length))*float(n);
 uint ix=uint(u),iz=uint(v);float fx=u-float(ix),fz=v-float(iz);float4 result=make_float4(0.0f,0.0f,0.0f,0.0f);
 for(uint dz=0u;dz<2u;dz++)for(uint dx=0u;dx<2u;dx++){
  float weight=(dx==0u?1.0f-fx:fx)*(dz==0u?1.0f-fz:fz);float4 a=map[((iz+dz)%n)*n+(ix+dx)%n];
  result.x+=a.x*weight;result.y+=a.y*weight;result.z+=a.z*weight;result.w+=a.w*weight;
 }return result;
}
__device__ float2 sampleDisplacement(const float2* map,float x,float z,uint n,float length){
 float u=(x/length-floorf(x/length))*float(n),v=(z/length-floorf(z/length))*float(n);
 uint ix=uint(u),iz=uint(v);float fx=u-float(ix),fz=v-float(iz);float2 result=make_float2(0.0f,0.0f);
 for(uint dz=0u;dz<2u;dz++)for(uint dx=0u;dx<2u;dx++){
  float weight=(dx==0u?1.0f-fx:fx)*(dz==0u?1.0f-fz:fz);float2 a=map[((iz+dz)%n)*n+(ix+dx)%n];result.x+=a.x*weight;result.y+=a.y*weight;
 }return result;
}
__device__ float3 rippleBand(float x,float z,float time,float ripples,float cameraX,float cameraZ){
 float distance=sqrtf((cameraX-x)*(cameraX-x)+(cameraZ-z)*(cameraZ-z));
 float detail=ripples*expf(-distance*0.018f),height=0.0f,dx=0.0f,dz=0.0f;
 for(uint k=0u;k<4u;k++){
  float angle=0.6f+float(k)*1.37f,klen=0.9f+float(k)*0.65f;
  float kx=cosf(angle)*klen,kz=sinf(angle)*klen,w=sqrtf(9.81f*klen+0.000074f*klen*klen*klen);
  float phase=kx*x+kz*z-w*time+float(k)*2.3f,amp=detail*0.023f/(1.0f+float(k)*0.8f);
  height+=amp*sinf(phase);dx+=amp*kx*cosf(phase);dz+=amp*kz*cosf(phase);
 }return make_float3(height,dx,dz);
}
__global__ void cloud_splats(float4* centers,float4* covA,float4* covB,uint* colors,const float4* volume,
 uint offset,uint count,uint cx,uint cy,uint cz,float time,float wind,float warmth,float cameraX,float cameraY,float cameraZ,uint enabled){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;uint j=offset+i;
 uint ix=i%cx,iy=(i/cx)%cy,iz=i/(cx*cy);
 float ux=(float(ix)+0.15f+vhash(i+7u)*0.7f)/float(cx)+time*wind*4.5f/480.0f;
 float uz=(float(iz)+0.15f+vhash(i+41u)*0.7f)/float(cz)+time*wind*1.4f/480.0f;
 float x=-240.0f+(ux-floorf(ux))*480.0f,y=12.0f+(float(iy)+0.15f+vhash(i+17u)*0.7f)*138.0f/float(cy),z=-240.0f+(uz-floorf(uz))*480.0f;
 float4 f=sampleVolume(volume,x,y,z);f.x=densityAt(x,y,z,time,wind);
 float vx=x-cameraX,vy=y-cameraY,vz=z-cameraZ,inv=rsqrtf(vx*vx+vy*vy+vz*vz);
 float mu=(vx*(-0.22f)+vy*0.19f+vz*(-0.957f))*inv;
 float3 light=cloudRadiance(f.y,f.z,mu,warmth);
 float size=480.0f/float(cx)*0.50f,sy=138.0f/float(cy)*0.52f,sz=480.0f/float(cz)*0.50f;
 float alpha=1.0f-expf(-f.x*size*1.6f);
 centers[j]=make_float4(x,y,z,1.0f);covA[j]=make_float4(size*size,0.0f,0.0f,sy*sy);covB[j]=make_float4(0.0f,sz*sz,0.0f,0.0f);
 colors[j]=rgba(light.x,light.y,light.z,enabled!=0u?alpha:0.0f);
}
__global__ void atmosphere_splats(float4* centers,float4* covA,float4* covB,uint* colors,uint offset,uint count,float warmth){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;uint j=offset+i;
 float longitude=(float(i%64u)+0.5f)/64.0f*6.28318530718f,elevation=-0.28f+float(i/64u)/31.0f*1.83f;
 float dx=sinf(longitude)*cosf(elevation),dy=sinf(elevation),dz=-cosf(longitude)*cosf(elevation);
 float3 col=skyColor(fmaxf(0.0f,dy),warmth);
 float glow=powf(fmaxf(0.0f,dx*(-0.22f)+dy*0.19f+dz*(-0.957f)),32.0f);
 centers[j]=make_float4(dx*4400.0f,dy*4400.0f,dz*4400.0f,2.0f);covA[j]=make_float4(130000.0f,0.0f,0.0f,130000.0f);covB[j]=make_float4(0.0f,130000.0f,0.0f,0.0f);
 colors[j]=rgba(col.x+glow*0.42f,col.y+glow*0.15f,col.z+glow*0.015f,0.19f);
}
__global__ void ray_splats(float4* centers,float4* covA,float4* covB,uint* colors,const float4* volume,
 uint offset,uint count,float time,float strength,uint enabled,uint cloudEnabled){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;uint j=offset+i;
 float x=-180.0f+360.0f*vhash(i+381u),y=2.0f+30.0f*vhash(i+721u),z=-180.0f+360.0f*vhash(i+191u);
 float optical=0.0f;
 if(cloudEnabled!=0u)for(uint s=0u;s<16u;s++){float t=(float(s)+0.5f)*6.0f;optical+=sampleVolume(volume,x-0.22f*t,y+0.19f*t,z-0.957f*t).x*6.0f;}
 float light=expf(-optical),alpha=strength*0.048f*light*expf(-y*0.06f);
 centers[j]=make_float4(x,y,z,3.0f);covA[j]=make_float4(5.0f,0.0f,0.0f,2.0f);covB[j]=make_float4(-2.5f,14.0f,0.0f,0.0f);
 colors[j]=rgba(0.95f,0.68f,0.35f,enabled!=0u?alpha:0.0f);
}
__global__ void sun_splats(float4* centers,float4* covA,float4* covB,uint* colors,uint offset,uint count){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;uint j=offset+i;float s=i==0u?27.0f:38.0f+float(i)*7.0f;
 centers[j]=make_float4(-770.0f,665.0f,-3350.0f,4.0f);covA[j]=make_float4(s*s,0.0f,0.0f,s*s);covB[j]=make_float4(0.0f,0.2f,0.0f,0.0f);
 colors[j]=rgba(1.0f,0.81f,0.45f,i==0u?1.0f:0.023f);
}
