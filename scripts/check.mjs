import { readFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { compile } from '../vendor/cuda-webshader/src/compiler/compiler.js';

const root = new URL('../', import.meta.url);
for (const directory of ['', 'scripts/', 'tests/']) {
  for (const name of await readdir(new URL(directory, root))) {
    if (!/\.(js|mjs)$/.test(name)) continue;
    const result = spawnSync(process.execPath, ['--check', fileURLToPath(new URL(directory + name, root))], { stdio: 'inherit' });
    if (result.status !== 0) throw Error(`JavaScript syntax check failed: ${name}`);
  }
}
const source = (await Promise.all(['volume.cu', 'scene.cu', 'caustics.cu', 'water.cu', 'surface-splats.cu', 'ocean-fft.cu'].map(name => readFile(new URL(name, root), 'utf8')))).join('\n');
const entries = [...source.matchAll(/__global__\s+void\s+(\w+)\s*\(/g)].map(match => match[1]);
for (const entry of entries) {
  const artifact = compile(source, { entry, workgroupSize: [128, 1, 1] });
  if (!artifact.wgsl?.includes('@compute')) throw Error(`No compute shader emitted for ${entry}`);
  console.log(`Compiled ${entry}`);
}
const provenance = JSON.parse(await readFile(new URL('vendor/cuda-webshader/PROVENANCE.json', root), 'utf8'));
for (const { path, sha256 } of provenance.files) {
  const data = await readFile(new URL(`vendor/cuda-webshader/${path}`, root));
  if (createHash('sha256').update(data).digest('hex') !== sha256) throw Error(`Vendored source changed without updating provenance: ${path}`);
}
console.log(`Passed syntax checks, ${entries.length} CUDA kernel compilations, and ${provenance.files.length} vendored source hashes.`);
