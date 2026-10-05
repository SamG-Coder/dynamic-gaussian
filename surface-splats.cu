// World-space Gaussian water and submerged sand. No pixel buffer or screen quad.
// Both representations share the same FFT field, reflected volume and sand illumination.
__global__ void surface_splats(float4* centers,float4* covA,float4* covB,uint* colors,
 const float4* oceanMap,const float2* displacement,const float4* volume,const uint* caustics,
 uint offset,uint nx,uint nz,uint farfield,uint seabed,uint n,float length,float time,float swell,float ripples,float warmth,float reflections,
 float cameraX,float cameraY,float cameraZ,float depth,float causticStrength,uint resolution,uint enabled,uint cloudEnabled,uint sandEnabled){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nx*nz)return;uint j=offset+i;
 float u=(float(i%nx)+0.5f)/float(nx),v=(float(i/nx)+0.5f)/float(nz);
 float angle=u*6.28318530718f,ct=cosf(angle),st=sinf(angle);
 float radius=72.0f/fmaxf(fabsf(ct),fabsf(st))+v*v*v*5900.0f;
 float x=cameraX+(farfield!=0u?ct*radius:(u-0.5f)*160.0f);
 float z=cameraZ+(farfield!=0u?st*radius:(v-0.5f)*160.0f);
 float edge=fmaxf(fabsf(x-cameraX),fabsf(z-cameraZ));
 float blend=farfield!=0u?sat((edge-72.0f)/8.0f):sat((80.0f-edge)/8.0f);
 float4 wave=waterField(oceanMap,displacement,x,z,n,length,time,swell,ripples,cameraX,cameraZ);
 float y=wave.x,dx=wave.y,dz=wave.z;
 float3 radiance=make_float3(0.0f,0.0f,0.0f);
 if(seabed!=0u){
  y=-bottomDepth(x,z,depth);
  dx=-(bottomDepth(x+0.1f,z,depth)-bottomDepth(x-0.1f,z,depth))*5.0f;
  dz=-(bottomDepth(x,z+0.1f,depth)-bottomDepth(x,z-0.1f,depth))*5.0f;
  if(enabled!=0u)radiance=sandRadiance(caustics,x,z,depth,causticStrength,resolution,length,warmth,cameraX,cameraZ);
 }else if(enabled!=0u){
  radiance=waterRadiance(volume,caustics,x,y,z,wave,swell,warmth,reflections,cameraX,cameraY,cameraZ,depth,causticStrength,resolution,length,cloudEnabled,sandEnabled);
 }
 // Tangent-aligned positive-definite covariance; adaptive far rings overlap the near grid.
 float sx=farfield!=0u?radius*6.28318530718f/float(nx)*0.72f:160.0f/float(nx)*0.72f;
 float sz=farfield!=0u?fmaxf(0.25f,17700.0f*v*v/float(nz)*0.72f):160.0f/float(nz)*0.72f;
 float xx=farfield!=0u?sx*sx*st*st+sz*sz*ct*ct:sx*sx;
 float zz=farfield!=0u?sx*sx*ct*ct+sz*sz*st*st:sz*sz;
 float xz=farfield!=0u?(sz*sz-sx*sx)*ct*st:0.0f;
 centers[j]=make_float4(x,y,z,seabed!=0u?6.0f:5.0f);
 covA[j]=make_float4(xx,xx*dx+xz*dz,xz,xx*dx*dx+2.0f*xz*dx*dz+zz*dz*dz+0.002f);
 covB[j]=make_float4(xz*dx+zz*dz,zz,0.0f,0.0f);
 colors[j]=rgba(radiance.x,radiance.y,radiance.z,enabled!=0u?0.995f*blend:0.0f);
}
