#map = affine_map<(d0) -> (d0)>
module  {
  func @forward(%arg0: tensor<4xf16>, %arg1: tensor<4xf16>) -> tensor<4xf16> {
    // call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
    %float = arith.constant 0.324 :f32
    %float_result = math.tanh %float : f32
    %c0 = arith.constant 0 : index
    // %dim = tensor.dim %arg0, %c0 : tensor<4xf16>
    %0 = linalg.init_tensor [4] : tensor<4xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<4xf16>, tensor<4xf16>) outs(%0 : tensor<4xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %5 = arith.addf %arg2, %arg3 : f16
      linalg.yield %5 : f16
    } -> tensor<4xf16>
    %2 = tensor.cast %1 : tensor<4xf16> to tensor<4xf16>
    return %2 : tensor<4xf16>
  }

 func.func private @mlirAsyncRuntimePrintCurrentThreadId() -> () attributes { llvm.emit_c_interface }
}
