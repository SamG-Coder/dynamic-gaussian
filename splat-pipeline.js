import * as THREE from 'three/webgpu';
import { instanceIndex, vec4, uint, max, log2 } from 'three/tsl';
import { GaussianSplat } from 'three/addons/objects/GaussianSplat.js';
import { createGaussianSplatGeometry } from 'three/addons/utils/GaussianSplatUtils.js';

/** Pinned r186 adapter. This file alone accesses GaussianSplat's private buffers.
 * All layers share a single depth sort. CPU arrays are allocation seeds only.
 */
export class CudaSplatPipeline {
  static async create({ renderer, runtime, source, layers, simulation }) {
    if (THREE.REVISION !== '186' || renderer.backend.device !== runtime.device || !renderer.backend.isWebGPUBackend)
      throw new Error('This adapter requires Three r186 on the same WebGPU device as CUDA WebShader.');
    const pipeline = new CudaSplatPipeline();
    Object.assign(pipeline, { renderer, runtime, layers, simulation, resources: {}, attributes: [], passes: [], disposed: false });
    let offset = 0;
    for (const layer of layers) { layer.offset = offset; offset += layer.count; }
    pipeline.count = offset;
    const geometry = createGaussianSplatGeometry(new Float32Array(offset * 3), new Float32Array(offset * 6), new Uint8Array(offset * 4));
    const mesh = pipeline.mesh = new GaussianSplat(geometry, { autoSort: false });
    mesh.name = 'CUDA procedural Gaussian scene';
    // Conservative authored bounds, never infer bounds from stale CPU allocation seeds.
    mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 1000, -2000), 9000);
    mesh.boundingBox = new THREE.Box3(new THREE.Vector3(-7500, -1000, -4800), new THREE.Vector3(7500, 4300, 65));
    mesh.frustumCulled = false;
    // Default raycasting reads CPU seed geometry. Disable it until GPU picking exists.
    mesh.raycast = () => {};
    // Logarithmic bins preserve near-water order while also sorting a 4 km backdrop.
    mesh._sort.setBinNode(() => {
      const center=mesh._buffers.centerRead.element(instanceIndex).xyz;
      const depth=mesh._sortMatrix.mul(vec4(center,1)).z.negate();
      const bin=uint(log2(max(depth,.25).div(.25)).div(Math.log2(6000/.25)).clamp(0,1).mul(4095));
      return uint(4095).sub(bin);
    });
    for (const [name, key] of Object.entries({ centers: 'centerRead', covA: 'covarianceARead', covB: 'covarianceBRead', colors: 'colorRead' })) {
      const attribute = mesh._buffers[key].value;
      renderer.backend.createStorageAttribute(attribute);
      const buffer = renderer.backend.get(attribute).buffer;
      if (!buffer) throw new Error(`Missing splat buffer ${name}`);
      pipeline.resources[name] = runtime.importBuffer(buffer, attribute.array.byteLength, `splat ${name}`);
      pipeline.attributes.push(attribute);
    }
    try {
      for (const layer of layers) {
        const kernel = await runtime.kernel(source, { entry: layer.entry, workgroupSize: [128, 1, 1] });
        const values = layer.scalars({ time: 0, camera: { x: 0, y: 12, z: 42 } });
        const available={...pipeline.resources,...simulation?.resources};
        const bindings=Object.fromEntries(kernel.artifact.metadata.bindings.map(binding=>[binding.name,available[binding.name]]));
        pipeline.passes.push({ layer, kernel, invocation: kernel.bind(bindings, { offset: layer.offset, ...values }) });
      }
    } catch (error) { pipeline.dispose(); throw error; }
    return pipeline;
  }
  update(frame, camera) {
    if (this.disposed) throw new Error('Splat pipeline disposed');
    const batch = this.runtime.batch({ label: 'CUDA Gaussian scene generation' });
    this.simulation?.record(batch,frame);
    for (const { layer, invocation } of this.passes) {
      invocation.setScalars({ offset: layer.offset, ...layer.scalars(frame) });
      batch.dispatch(invocation, [Math.ceil(layer.count / 128), 1, 1]);
    }
    batch.submit();
    camera.updateMatrixWorld();
    // Positions change even when the camera does not: invalidate direction-only caching.
    this.mesh._sortInitialized = false;
    this.mesh.updateSort(this.renderer, camera);
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    for (const resource of Object.values(this.resources)) this.runtime.destroyBuffer(resource);
    this.mesh.dispose();
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.mesh.splatGeometry.dispose();
    for (const attribute of this.attributes) this.renderer.backend.destroyAttribute(attribute);
  }
}
