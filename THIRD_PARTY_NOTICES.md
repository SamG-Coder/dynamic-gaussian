# Third-party notices

Dynamic Gaussian is licensed under MIT. See LICENSE.

- **CUDA WebShader compiler/runtime** — MIT, copyright 2026 SamG-Coder and CUDA WebShader contributors. Source: https://github.com/SamG-Coder/cuda-webshader. The exact upstream commit and SHA-256 for each included module are recorded in `vendor/cuda-webshader/PROVENANCE.json`; the original license is retained beside it. The standalone app explicitly selects WebGPU.
- **Three.js 0.186.0** — MIT, copyright Three.js authors. Source: https://github.com/mrdoob/three.js. Installed from the pinned npm dependency. The static build includes its original LICENSE under `vendor/three/LICENSE`. The Gaussian adapter depends on this revision's internal storage/sorting API.
- **Playwright 1.56.1** — Apache-2.0, Microsoft Corporation. Development/test dependency; not shipped in the website.
- **Acorn 8.15.0** — MIT, Acorn contributors. Build-time JavaScript parser used to validate staged imports; not shipped in the website.

The spectral FFT implementation was adapted from the MIT-licensed CUDA WebShader implementation. No captured splat assets, third-party photographs, or proprietary textures are included.
