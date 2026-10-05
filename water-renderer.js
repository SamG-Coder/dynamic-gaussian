import * as THREE from 'three/webgpu';
import { storage, uint, uniform, screenCoordinate, positionGeometry, vec4, unpackUnorm4x8 } from 'three/tsl';

/** CUDA pixels on the renderer's device; TSL only presents the shared buffer. */
export class CudaWaterRenderer {
  static async create(renderer, runtime, simulation, source) {
    const water = new CudaWaterRenderer();
    Object.assign(water, { renderer, runtime, simulation, size: new THREE.Vector2(), widthNode: uniform(1, 'uint') });
    water.kernel = await runtime.kernel(source, { entry: 'water_pixels', workgroupSize: [128, 1, 1] });
    water.geometry = new THREE.PlaneGeometry(2, 2);
    water.material = new THREE.NodeMaterial();
    water.material.transparent = true;water.material.depthTest = false;water.material.depthWrite = false;
    water.material.vertexNode = vec4(positionGeometry.xy, 0, 1);
    water.mesh = new THREE.Mesh(water.geometry, water.material);
    water.mesh.frustumCulled = false; water.mesh.renderOrder = 1000;
    return water;
  }
  resize() {
    this.renderer.getDrawingBufferSize(this.size);
    const width=this.size.x, height=this.size.y;
    if(width===this.width && height===this.height)return;
    if(this.resource){this.runtime.destroyBuffer(this.resource);this.renderer.backend.destroyAttribute(this.attribute);}
    Object.assign(this,{width,height});this.widthNode.value=width;
    this.attribute=new THREE.StorageBufferAttribute(new Uint32Array(width*height),1);
    this.renderer.backend.createStorageAttribute(this.attribute);
    this.resource=this.runtime.importBuffer(this.renderer.backend.get(this.attribute).buffer,width*height*4,'CUDA water pixels');
    const pixels=storage(this.attribute,'uint',width*height).toReadOnly();
    const index=uint(screenCoordinate.y).mul(this.widthNode).add(uint(screenCoordinate.x));
    const color=unpackUnorm4x8(pixels.element(index));
    this.material.colorNode=color.rgb;this.material.opacityNode=color.a;this.material.needsUpdate=true;
    this.invocation=null;
  }
  update(time,camera,state){
    this.resize();camera.updateMatrixWorld();const e=camera.matrixWorld.elements,s=this.simulation;
    const scalars={width:this.width,height:this.height,n:s.n,length:s.length,time,swell:state.swell,ripples:state.ripples,warmth:state.warmth,reflections:state.reflections,
      cameraX:e[12],cameraY:e[13],cameraZ:e[14],rightX:e[0],rightY:e[1],rightZ:e[2],upX:e[4],upY:e[5],upZ:e[6],forwardX:-e[8],forwardY:-e[9],forwardZ:-e[10],tangent:Math.tan(camera.fov*Math.PI/360),aspect:camera.aspect,
      depth:state.depth,causticStrength:state.causticStrength,resolution:s.resolution,enabled:+state.ocean,cloudEnabled:+state.clouds,sandEnabled:+state.sand,farEnabled:+state.far};
    if(!this.invocation){const available={pixels:this.resource,...s.resources};const bindings=Object.fromEntries(this.kernel.artifact.metadata.bindings.map(b=>[b.name,available[b.name]]));this.invocation=this.kernel.bind(bindings,scalars);}
    this.runtime.batch().dispatch(this.invocation.setScalars(scalars),[Math.ceil(this.width*this.height/128)]).submit();
  }
  dispose(){if(this.resource){this.runtime.destroyBuffer(this.resource);this.renderer.backend.destroyAttribute(this.attribute);}this.geometry.dispose();this.material.dispose();}
}
