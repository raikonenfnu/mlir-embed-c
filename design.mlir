// Output MLIR
module {
  func private @OpName() attributes {llvm.emit_c_interface}
  func @forward_dispatch_0() {
    call @OpName() : () -> ()
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<4xf16>
    memref.assume_alignment %0, 64 : memref<4xf16>
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:4xf16>
    %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<4xf16>
    memref.assume_alignment %2, 64 : memref<4xf16>
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:4xf16>
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : memref<4xf16>
    memref.assume_alignment %4, 64 : memref<4xf16>
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:4xf16>
    %tensorA = call @SliceTensorPerPE(%0) : memref<4xf16> -> memref<?xf16>
    %tensorB = call @SliceTensorPerPE(%2) : memref<4xf16> -> memref<?xf16>
    %tensorC = call @SliceTensorPerPE(%4) : memref<4xf16> -> memref<?xf16>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %6 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_x]
    %7 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_x]
    scf.for %arg0 = %6 to %c4 step %7 {
      // %8 = memref.subview %4[%arg0] [4] [1] : memref<4xf16> to memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>
      // %9 = memref.subview %0[%arg0] [4] [1] : memref<4xf16> to memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>
      // %10 = memref.subview %2[%arg0] [4] [1] : memref<4xf16> to memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>
      %8 = call @mem_in %tensorC[%arg0] [4] [1] : memref<4xf16> to memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>
      %9 = call @mem_in %tensorA[%arg0] [4] [1] : memref<4xf16> to memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>
      %10 = call @mem_in %tensorB[%arg0] [4] [1] : memref<4xf16> to memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>
      %11 = vector.transfer_read %9[%c0], %cst {in_bounds = [true]} : memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>, vector<4xf16>
      %12 = vector.transfer_read %10[%c0], %cst {in_bounds = [true]} : memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>, vector<4xf16>
      %13 = arith.addf %11, %12 : vector<4xf16>
      vector.transfer_write %13, %8[%c0] {in_bounds = [true]} : vector<4xf16>, memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>
      %14 = call @mem_out %tensorC[%arg0] [4] [%c1] : memref<4xf16> to memref<4xf16, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%8 : memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>) outs(%14 : memref<4xf16, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>) {
      ^bb0(%arg1: f16, %arg2: f16):
        linalg.yield %arg1 : f16
      }
    }
    return
  }
}

// Input MLIR
module {
  func @forward_dispatch_0() {
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<4xf16>
    memref.assume_alignment %0, 64 : memref<4xf16>
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:4xf16>
    %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<4xf16>
    memref.assume_alignment %2, 64 : memref<4xf16>
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:4xf16>
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : memref<4xf16>
    memref.assume_alignment %4, 64 : memref<4xf16>
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:4xf16>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %6 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_x]
    %7 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_x]
    scf.for %arg0 = %6 to %c4 step %7 {
      %8 = memref.subview %4[%arg0] [4] [1] : memref<4xf16> to memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>
      %9 = memref.subview %0[%arg0] [4] [1] : memref<4xf16> to memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>
      %10 = memref.subview %2[%arg0] [4] [1] : memref<4xf16> to memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>
      %11 = vector.transfer_read %9[%c0], %cst {in_bounds = [true]} : memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>, vector<4xf16>
      %12 = vector.transfer_read %10[%c0], %cst {in_bounds = [true]} : memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>, vector<4xf16>
      %13 = arith.addf %11, %12 : vector<4xf16>
      vector.transfer_write %13, %8[%c0] {in_bounds = [true]} : vector<4xf16>, memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>
      %14 = memref.subview %4[%arg0] [4] [%c1] : memref<4xf16> to memref<4xf16, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%8 : memref<4xf16, affine_map<(d0)[s0] -> (d0 + s0)>>) outs(%14 : memref<4xf16, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>) {
      ^bb0(%arg1: f16, %arg2: f16):
        linalg.yield %arg1 : f16
      }
    }
    return
  }
}
