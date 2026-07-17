# Third-Party Notices

## GaussianGPT source

This extension vendors the complete tracked tree of:

- **Project:** GaussianGPT
- **Repository:** https://github.com/nicolasvonluetzow/GaussianGPT
- **Commit:** `0615470a5ff359c676408e9ae42036d534d04a43`
- **License:** MIT
- **Copyright:** Copyright (c) 2026 Nicolas von Luetzow
- **Location:** `vendor/GaussianGPT`
- **License copy:** `vendor/GaussianGPT/LICENSE`

The upstream source is included by `git archive`; Git metadata, checkpoints,
datasets, logs, and generated outputs are not included. `UPSTREAM.json`
records the immutable revision and vendored inventory.

GaussianGPT acknowledges gsplat, MinkowskiEngine, Flash-Attention,
vector-quantize-pytorch, and nanochat. Those packages are dependencies or
acknowledged upstream influences; they are not vendored by this extension.
Their own license terms apply when installed.

Setup builds the documented MinkowskiEngine compatibility source from
immutable commit `1a17f71f3158b9e94e90703961695de627f3df08`. On the CUDA 13
lane it applies the deterministic `gaussiangpt-cuda13-compat-v1` compatibility
patch set, builds the wheel against the extension-local static OpenBLAS CBLAS
archive, and caches the validated wheel by its full build identity. The Linux
aarch64 CPython 3.11.9, CUDA 13.0, SM 121 path passed the required native CUDA
probes on 2026-07-14. CUDA 12.8 remains unvalidated.

## OpenBLAS static CBLAS dependency

Setup builds and caches the following dependency inside the extension:

- **Project:** OpenBLAS
- **Repository:** https://github.com/OpenMathLib/OpenBLAS
- **Version:** 0.3.26
- **Commit:** `6c77e5e314474773a7749357b153caba4ec3817d`
- **License:** BSD-3-Clause
- **Use:** statically linked CBLAS implementation used by the cached
  MinkowskiEngine wheel

The OpenBLAS license notice follows in full:

```text
Copyright (c) 2011-2014, The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
   3. Neither the name of the OpenBLAS project nor the names of 
      its contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Model checkpoints

The VQ-VAE and GPT checkpoints are distributed by the GaussianGPT authors from
`https://kaldir.vc.cit.tum.de/gaussiangpt/`. The official repository and the
checkpoint-host README publish filenames and SHA256 values but do **not**
declare model-weight license terms. Their license is therefore recorded as
**unknown**. The extension license and upstream code license do not imply a
license grant for the checkpoints.

## Datasets

No 3D-FRONT, ASE, PhotoShape, or other dataset content is included. Dataset
licenses are independent of this extension, its vendored source, and the
checkpoint-license status.
