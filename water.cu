// Full-resolution water shading, computed in CUDA and presented by a Three TSL quad.
// One continuous ray/height-field surface covers near water through the horizon.
__device__ float4 waterField(const float4* oceanMap,const float2* displacement,float x,float z,uint n,float length,float time,float swell,float ripples,float cameraX,float cameraZ){
 float radial=sqrtf((x-cameraX)*(x-cameraX)+(z-cameraZ)*(z-cameraZ));
 float fade=expf(-fmaxf(0.0f,radial-160.0f)*0.0015f);
 float2 shift=sampleDisplacement(displacement,x,z,n,length);
 float4 w=sampleOcean(oceanMap,x-shift.x*swell*fade,z-shift.y*swell*fade,n,length);
 float3 r=rippleBand(x,z,time,ripples,cameraX,cameraZ);
 return make_float4(w.x*swell*fade+r.x,w.y*swell*fade+r.y,w.z*swell*fade+r.z,w.w);
}
__global__ void water_pixels(uint* pixels,const float4* oceanMap,const float2* displacement,const float4* volume,const uint* caustics,
 uint width,uint height,uint n,float length,float time,float swell,float ripples,float warmth,float reflections,
 float cameraX,float cameraY,float cameraZ,float rightX,float rightY,float rightZ,float upX,float upY,float upZ,float forwardX,float forwardY,float forwardZ,float tangent,float aspect,
 float depth,float causticStrength,uint resolution,uint enabled,uint cloudEnabled,uint sandEnabled,uint farEnabled){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width*height)return;
 float u=(2.0f*(float(i%width)+0.5f)/float(width)-1.0f)*tangent*aspect;
 float v=(1.0f-2.0f*(float(i/width)+0.5f)/float(height))*tangent;
 float rayX=forwardX+u*rightX+v*upX,rayY=forwardY+u*rightY+v*upY,rayZ=forwardZ+u*rightZ+v*upZ;
 float ri=rsqrtf(rayX*rayX+rayY*rayY+rayZ*rayZ);rayX*=ri;rayY*=ri;rayZ*=ri;
 pixels[i]=0u;if(rayY>=-0.001f)return;
 float planeT=-cameraY/rayY;if(planeT>6000.0f)return;
 float t=planeT;
 if(enabled!=0u){
  // Bracket the closest visible crossing, then refine it. The wave envelope fades at distance.
  float amplitude=4.0f*swell+0.15f;
  float start=fmaxf(0.0f,(cameraY-amplitude)/(-rayY)),end=fminf(6000.0f,(cameraY+amplitude)/(-rayY));
  float previous=start,hit=end;uint found=0u;
  for(uint s=1u;s<=24u;s++){
   float q=start+(end-start)*float(s)/24.0f;
   float4 w=waterField(oceanMap,displacement,cameraX+rayX*q,cameraZ+rayZ*q,n,length,time,swell,ripples,cameraX,cameraZ);
   if(cameraY+rayY*q<=w.x){hit=q;found=1u;break;}previous=q;
  }
  if(found==0u)return;
  for(uint s=0u;s<7u;s++){
   float q=(previous+hit)*0.5f;float4 w=waterField(oceanMap,displacement,cameraX+rayX*q,cameraZ+rayZ*q,n,length,time,swell,ripples,cameraX,cameraZ);
   if(cameraY+rayY*q>w.x)previous=q;else hit=q;
  }t=(previous+hit)*0.5f;
 }else{
  if(sandEnabled==0u)return;
  for(uint s=0u;s<4u;s++)t=(-bottomDepth(cameraX+rayX*t,cameraZ+rayZ*t,depth)-cameraY)/rayY;
 }
 float x=cameraX+rayX*t,z=cameraZ+rayZ*t,y=cameraY+rayY*t;
 float radial=sqrtf((x-cameraX)*(x-cameraX)+(z-cameraZ)*(z-cameraZ));
 if(farEnabled==0u&&radial>80.0f)return;
 if(enabled==0u){float3 sand=sandRadiance(caustics,x,z,depth,causticStrength,resolution,length,warmth,cameraX,cameraZ);pixels[i]=rgba(sand.x,sand.y,sand.z,1.0f);return;}
 float4 wave=waterField(oceanMap,displacement,x,z,n,length,time,swell,ripples,cameraX,cameraZ);
 float dx=wave.y,dz=wave.z;
 float inv=rsqrtf(1.0f+dx*dx+dz*dz),normalX=-dx*inv,normalY=inv,normalZ=-dz*inv;
 float vx=cameraX-x,vy=cameraY-y,vz=cameraZ-z,il=rsqrtf(vx*vx+vy*vy+vz*vz);vx*=il;vy*=il;vz*=il;
 float ndv=fmaxf(0.0f,normalX*vx+normalY*vy+normalZ*vz);
 float rx=2.0f*ndv*normalX-vx,ry=2.0f*ndv*normalY-vy,rz=2.0f*ndv*normalZ-vz;
 float3 env=skyColor(fmaxf(0.0f,ry),warmth);
 float4 cloud=make_float4(0.0f,0.0f,0.0f,1.0f);
 if(cloudEnabled!=0u&&reflections>0.0f&&radial<420.0f){
  cloud=traceCloud(volume,x,y+0.2f,z,rx,ry,rz,warmth);
  float blend=sat((420.0f-radial)/160.0f);cloud.x*=blend;cloud.y*=blend;cloud.z*=blend;cloud.w=1.0f-(1.0f-cloud.w)*blend;
 }
 env.x=env.x*cloud.w+cloud.x;env.y=env.y*cloud.w+cloud.y;env.z=env.z*cloud.w+cloud.z;
 float sunDot=fmaxf(0.0f,rx*(-0.22f)+ry*0.19f+rz*(-0.957f));
 float glint=(powf(sunDot,320.0f)*2.8f+powf(sunDot,38.0f)*0.08f)*cloud.w;
 float fresnel=0.0204f+0.9796f*powf(1.0f-ndv,5.0f),reflect=reflections*fresnel;
 // Refract the view ray onto the sand bed. Only this submerged hit samples caustics.
 float eta=0.7501875f,refractK=eta*ndv-sqrtf(1.0f-eta*eta*(1.0f-ndv*ndv));
 float tx=-eta*vx+refractK*normalX,ty=-eta*vy+refractK*normalY,tz=-eta*vz+refractK*normalZ;
 float travel=(-bottomDepth(x,z,depth)-y)/fminf(-0.02f,ty);
 for(uint s=0u;s<3u;s++)travel=(-bottomDepth(x+tx*travel,z+tz*travel,depth)-y)/fminf(-0.02f,ty);
 travel=fmaxf(0.0f,travel);
 float3 sand=sandRadiance(caustics,x+tx*travel,z+tz*travel,depth,causticStrength,resolution,length,warmth,cameraX,cameraZ);
 if(sandEnabled==0u)sand=make_float3(0.0f,0.0f,0.0f);
 float tr=expf(-0.18f*travel),tg=expf(-0.075f*travel),tb=expf(-0.035f*travel);
 float underR=sand.x*tr+0.008f*(1.0f-tr),underG=sand.y*tg+0.066f*(1.0f-tg),underB=sand.z*tb+0.085f*(1.0f-tb);
 float foam=sat(wave.w*1.4f)*swell*0.10f;
 float red=underR*(1.0f-reflect)+env.x*reflect+glint*reflections+foam;
 float green=underG*(1.0f-reflect)+env.y*reflect+glint*0.63f*reflections+foam;
 float blue=underB*(1.0f-reflect)+env.z*reflect+glint*0.27f*reflections+foam;
 float haze=1.0f-expf(-radial*0.0006f);float3 horizon=skyColor(0.0f,warmth);
 red=red*(1.0f-haze)+horizon.x*haze;green=green*(1.0f-haze)+horizon.y*haze;blue=blue*(1.0f-haze)+horizon.z*haze;

 pixels[i]=rgba(red,green,blue,1.0f);
}
