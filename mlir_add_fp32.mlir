#map = affine_map<(d0) -> (d0)>
module  {
  func @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    // call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
    %float = arith.constant 0.324 :f32
    %float_result = math.tanh %float : f32
    %c0 = arith.constant 0 : index
    // %dim = tensor.dim %arg0, %c0 : tensor<4xf32>
    %0 = linalg.init_tensor [4] : tensor<4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%0 : tensor<4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %5 = arith.addf %arg2, %arg3 : f32
      linalg.yield %5 : f32
    } -> tensor<4xf32>
    %2 = tensor.cast %1 : tensor<4xf32> to tensor<4xf32>
    return %2 : tensor<4xf32>
  }

 func.func private @mlirAsyncRuntimePrintCurrentThreadId() -> () attributes { llvm.emit_c_interface }
}
