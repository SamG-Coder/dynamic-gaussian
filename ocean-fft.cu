// SPDX-License-Identifier: MIT
// Phillips wind spectrum, deep-water dispersion, Hermitian evolution, three IFFTs.
__device__ float random01(uint n){n^=n>>16;n*=2146121005u;n^=n>>15;n*=2221713035u;n^=n>>16;return float(n&16777215u)/16777215.0f;}
__device__ float2 initialSpectrum(uint x,uint z,uint n,float length,float windSpeed){
 int ix=int(x),iz=int(z);if(x>n/2u)ix-=int(n);if(z>n/2u)iz-=int(n);
 float dk=6.28318530718f/length,kx=float(ix)*dk,kz=float(iz)*dk;
 float k2=kx*kx+kz*kz;if(k2<0.000001f)return make_float2(0.0f,0.0f);
 float L=windSpeed*windSpeed/9.81f,dir=(kx*0.8f+kz*0.6f)*rsqrtf(k2);
 float phillips=0.00065f*expf(-1.0f/(k2*L*L))*dir*dir*expf(-k2*0.12f)/(k2*k2);
 if(dir<0.0f)phillips*=0.18f;
 uint seed=x+z*n;
 float radius=sqrtf(-2.0f*logf(fmaxf(0.00001f,random01(seed*17u+123u))));
 float phase=6.28318530718f*random01(seed*31u+67u);
 // IFFT is normalized below; Fourier coefficients include N^2 and bin width.
 float scale=sqrtf(phillips*0.5f)*dk*float(n*n);
 return make_float2(radius*cosf(phase)*scale,radius*sinf(phase)*scale);
}
__global__ void evolve_spectrum(float2* spectrum,uint n,float length,float time,float windSpeed){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*n)return;
 uint x=i%n,z=i/n;int ix=int(x),iz=int(z);if(x>n/2u)ix-=int(n);if(z>n/2u)iz-=int(n);
 float kx=float(ix)*6.28318530718f/length,kz=float(iz)*6.28318530718f/length;
 float k=sqrtf(kx*kx+kz*kz),w=sqrtf(9.81f*k);
 float2 a=initialSpectrum(x,z,n,length,windSpeed),b=initialSpectrum((n-x)%n,(n-z)%n,n,length,windSpeed);
 float c=cosf(w*time),s=sinf(w*time);
 float2 h=make_float2((a.x+b.x)*c-(a.y+b.y)*s,(a.x-b.x)*s+(a.y-b.y)*c);
 spectrum[i]=h;
 float kdx=k>0.0f&&x!=n/2u?kx/k:0.0f,kdz=k>0.0f&&z!=n/2u?kz/k:0.0f;
 spectrum[n*n+i]=make_float2(h.y*kdx,-h.x*kdx);
 spectrum[2u*n*n+i]=make_float2(h.y*kdz,-h.x*kdz);
}
// Adapted from src/runtime/fft-kernels.js (MIT), bounded to N <= 256.
// Each workgroup transforms one row/column of one of three fields.
__global__ void inverse_fft(const float2* input,float2* output,uint n,uint axis){
 __shared__ float2 values[256];
 uint lane=threadIdx.x,line=blockIdx.x%n,field=blockIdx.x/n,base=field*n*n;
 for(uint i=lane;i<n;i+=blockDim.x){
  uint reverse=0u,bits=n;for(uint v=i;bits>1u;bits>>=1){reverse=(reverse<<1)|(v&1u);v>>=1;}
  uint index=axis==0u?line*n+i:i*n+line;values[reverse]=input[base+index];
 }
 __syncthreads();
 for(uint size=2u;size<=n;size<<=1){
  uint half=size>>1;
  for(uint butterfly=lane;butterfly<n/2u;butterfly+=blockDim.x){
   uint j=butterfly&(half-1u),start=(butterfly/half)*size;
   float angle=6.28318530718f*float(j)/float(size);
   float2 a=values[start+j],b=values[start+j+half];float c=cosf(angle),s=sinf(angle);
   float2 p=make_float2(b.x*c-b.y*s,b.x*s+b.y*c);
   values[start+j]=make_float2(a.x+p.x,a.y+p.y);values[start+j+half]=make_float2(a.x-p.x,a.y-p.y);
  }
  __syncthreads();
 }
 for(uint i=lane;i<n;i+=blockDim.x){uint index=axis==0u?line*n+i:i*n+line;output[base+index]=values[i];}
}
__global__ void ocean_surface(const float2* spatial,float4* oceanMap,float2* displacement,uint n,float length){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*n)return;
 uint x=i%n,z=i/n,l=z*n+(x+n-1u)%n,r=z*n+(x+1u)%n,b=((z+n-1u)%n)*n+x,t=((z+1u)%n)*n+x;
 float scale=1.0f/float(n*n),derivative=scale*float(n)/(2.0f*length);
 float sx=(spatial[r].x-spatial[l].x)*derivative,sz=(spatial[t].x-spatial[b].x)*derivative;
 float dxx=(spatial[n*n+r].x-spatial[n*n+l].x)*derivative;
 float dzz=(spatial[2u*n*n+t].x-spatial[2u*n*n+b].x)*derivative;
 float dxz=(spatial[n*n+t].x-spatial[n*n+b].x)*derivative;
 float jacobian=(1.0f-1.1f*dxx)*(1.0f-1.1f*dzz)-1.21f*dxz*dxz;
 oceanMap[i]=make_float4(spatial[i].x*scale,sx,sz,fmaxf(0.0f,0.5f-jacobian));
 displacement[i]=make_float2(-spatial[n*n+i].x*scale*1.1f,-spatial[2u*n*n+i].x*scale*1.1f);
}
