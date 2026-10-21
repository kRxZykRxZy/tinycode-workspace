# Hardware backends

The selection is capability based.  A backend is never chosen because its
name appears in a device enumeration: it has to initialise successfully
and pass a small execution test.

## Priority

1. **CUDA** – requires PyTorch with a usable CUDA device.
2. **Intel integrated graphics** – requires Intel graphics *and* a
   verified OpenVINO execution provider with the GPU plugin.
3. **ARM NEON** – requires an ARM CPU that reports NEON.
4. **CPU** – always available, the portable fallback.

## What each backend advertises

Every backend returns a `BackendCapabilities` object that lists the
operators, the precisions and the features it really supports.  The
artifact manifest declares the operators the model needs; the loader
refuses an artifact whose requirements the backend cannot satisfy.

## CUDA

`backends/cuda/` uses PyTorch.  It covers device discovery, capability
detection, model placement, GPU memory accounting, precision selection,
forward execution, synchronisation and out-of-memory reporting.  The
native CUDA path is optional; PyTorch already provides optimised kernels
for the operations that dominate the runtime.

The CUDA tests are skipped explicitly when no GPU is present.  A test
that did not run is never reported as passed.

## Intel integrated graphics

PyTorch does not support Intel graphics.  The backend therefore uses
OpenVINO:

1. the graphics adapters of the machine are enumerated,
2. the OpenVINO provider is probed **in a separate process**,
3. the GPU plugin has to be usable,
4. the model is compiled for the `GPU` device and the device that is
   actually used is reported.

**Why the probe runs in a subprocess:** on a machine whose OpenVINO GPU
plugin DLL is broken, *loading* the plugin aborts the whole process with
a Windows fatal exception (0xc0000139) instead of raising a Python
exception.  `provider.probe_gpu_plugin()` therefore runs the check in a
child process and reports a crash as a failed probe, so the server can
never be taken down by a broken driver.

**The model has to exist as an OpenVINO IR.**  The builder exports one
(`openvino` export in the builder, see `tinycode-model-builder`), and the
artifact manifest records it under `export.openvino`.  Without the IR the
backend reports itself as unavailable.

**The device is reported truthfully.**  After the compilation the backend
reads the `EXECUTION_DEVICES` property and reports the device that the
work really runs on.  When the execution lands on the CPU the backend
says so and does *not* claim GPU acceleration.

**Current limitation:** the exported graph has no KV cache, so the
OpenVINO path recomputes the whole sequence for every token.  It is a
compatibility path; the numpy runtime with the KV cache stays the default.

### What was verified on the development machine

| Check | Result |
| --- | --- |
| Graphics adapters enumerated | yes: `AMD Radeon HD 7500M/7600M Series` |
| Intel graphics present | **no** – this machine has no Intel adapter |
| OpenVINO installed | yes (2026.4.1) |
| OpenVINO devices | `['CPU']` only |
| GPU plugin loadable | **no** – `Cannot load library openvino_intel_gpu_plugin.dll: 127` |
| IR export → IR execution | yes, the logits match the PyTorch model within 1e-3 |
| Device reporting | the backend reports `CPU` and does not claim GPU |

So the OpenVINO execution path is implemented and verified end to end
(export, load, compile, infer, numerics, device reporting), but **the
Intel GPU acceleration itself could not be verified** on this machine:
there is no Intel adapter and the GPU plugin cannot be loaded.  The
backend correctly refuses to claim acceleration in that situation.


## ARM NEON

`backends/arm_neon/` uses the native kernels that are compiled with the
NEON code path.  The capability is detected at runtime; when NEON is not
available the portable CPU implementation is used.  ARM specific
instructions are never executed on another architecture.

## CPU

The CPU backend is a first class supported mode.  It uses numpy by
default and the native AVX2/NEON kernels when the shared library was
built.  Thread counts are configurable and the BLAS thread variables are
only set when the operator asks for it.

## Native library

```bash
cmake -S native -B native/build -DTINYCODE_BUILD_TESTS=ON
cmake --build native/build --config Release
```

The engine searches `native/build`, `native/build/lib`,
`src/tinycode_engine/runtime/_native` and `build/`, and honours
`TINYCODE_NATIVE_LIB`.
