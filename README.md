# MLIR Embed C examples
## _The first and only repo for injecting C/C++ code_

This repo is to show examples on how to inject C-API calls into MLIR.

## Instructions
### For C
```bash
iree-compile -iree-mlir-to-vm-bytecode-module -iree-hal-target-backends=dylib-llvm-aot --mlir-print-ir-after-all -iree-llvm-link-static --iree-codegen-inject-code=true --iree-memory-promotion-capi mlir_add_fp16.mlir -o /tmp/add.dylib 2> /tmp/dump.log
clang main.c forward_dispatch_0.s
./a.out
```
### For C++
```bash
iree-compile -iree-mlir-to-vm-bytecode-module -iree-hal-target-backends=dylib-llvm-aot --mlir-print-ir-after-all -iree-llvm-link-static --iree-codegen-inject-code=true --iree-memory-promotion-capi mlir_add_fp16.mlir -o /tmp/add.dylib 2> /tmp/dump.log
clang++ main.c forward_dispatch_0.s -I/path/to/llvm-project/mlir/include
./a.out
```
