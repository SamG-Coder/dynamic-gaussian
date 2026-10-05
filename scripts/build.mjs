import { readFile, writeFile, mkdir, cp, rm, readdir, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { parse } from 'acorn';

const root = fileURLToPath(new URL('../', import.meta.url)), out = path.resolve(root, 'dist');
if (path.dirname(out) !== path.resolve(root) || path.basename(out) !== 'dist') throw Error('Invalid build directory');
await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });
for (const file of await readdir(root)) {
  if (/\.(js|css|cu)$/.test(file) || ['LICENSE', 'THIRD_PARTY_NOTICES.md'].includes(file)) await cp(path.join(root, file), path.join(out, file));
}
await cp(path.join(root, 'vendor'), path.join(out, 'vendor'), { recursive: true });
for (const directory of ['build', 'examples/jsm']) {
  await cp(path.join(root, 'node_modules/three', directory), path.join(out, 'vendor/three', directory), { recursive: true });
}
await cp(path.join(root, 'node_modules/three/LICENSE'), path.join(out, 'vendor/three/LICENSE'));
const html = (await readFile(path.join(root, 'index.html'), 'utf8')).replaceAll('./node_modules/three/', './vendor/three/');
await writeFile(path.join(out, 'index.html'), html);
await writeFile(path.join(out, '.nojekyll'), '');

// Check the actual staged browser graph, including literal dynamic imports.
const imports = JSON.parse(html.match(/<script type="importmap">(.*?)<\/script>/s)[1]).imports;
function moduleSpecifiers(source) {
  const pending = [parse(source, { sourceType: 'module', ecmaVersion: 'latest' })], names = [];
  while (pending.length) {
    const node = pending.pop();
    if (['ImportDeclaration','ExportNamedDeclaration','ExportAllDeclaration','ImportExpression'].includes(node.type) && typeof node.source?.value === 'string') names.push(node.source.value);
    for (const value of Object.values(node)) {
      if (Array.isArray(value)) for (const item of value) { if (item?.type) pending.push(item); }
      else if (value?.type) pending.push(value);
    }
  }
  return names;
}
const queue = ['app.js']; const visited = new Set();
while (queue.length) {
  const file = queue.pop(); if (visited.has(file)) continue; visited.add(file);
  const text = await readFile(path.join(out, file), 'utf8');
  for (const name of moduleSpecifiers(text)) {
    let resolved;
    if (name.startsWith('.')) resolved = path.posix.normalize(path.posix.join(path.posix.dirname(file), name));
    else {
      const prefix = Object.keys(imports).filter(key => key === name || key.endsWith('/') && name.startsWith(key)).sort((a,b) => b.length-a.length)[0];
      if (!prefix) throw Error(`Unmapped browser import ${name} in ${file}`);
      resolved = imports[prefix].replace(/^\.\//, '') + name.slice(prefix.length);
    }
    if (resolved.startsWith('../') || resolved.startsWith('/')) throw Error(`Import escapes site: ${resolved}`);
    if (!(await stat(path.join(out, resolved))).isFile()) throw Error(`Missing import ${resolved}`);
    if (resolved.endsWith('.js')) queue.push(resolved);
  }
}
const manifest = {};
async function inventory(directory = '') {
  for (const entry of await readdir(path.join(out, directory), { withFileTypes: true })) {
    const file = path.posix.join(directory, entry.name);
    if (entry.isDirectory()) await inventory(file);
    else manifest[file] = createHash('sha256').update(await readFile(path.join(out, file))).digest('hex');
  }
}
await inventory();
await writeFile(path.join(out, 'asset-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(`Built dist: ${Object.keys(manifest).length} files; validated ${visited.size} reachable browser modules.`);
