module {
  func.func @main(%arg0: tensor<42x73x41x53xf32>, %arg1: tensor<70x100x95x59xf32>, %arg2: tensor<70xf32>, %arg3: tensor<84x61x52x56x47x17xi1>, %arg4: tensor<84x61x52x1x47x1xi1>, %arg5: tensor<27x55x98x30x85xi32>, %arg6: tensor<1x55x1x1x1xi32>, %arg7: tensor<24x59x61x2xi1>) -> (tensor<84x61x52x56x47x17xi1>, tensor<42x246x358x140xf32>, tensor<27x55x98x30x85xi1>, tensor<42x1x179x70xf32>, tensor<24x59x1x1xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 42, 246, 179, 70>} : (tensor<42x73x41x53xf32>, tensor<70x100x95x59xf32>, tensor<70xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<42x246x179x70xf32>
    %1 = tosa.logical_xor %arg3, %arg4 : (tensor<84x61x52x56x47x17xi1>, tensor<84x61x52x1x47x1xi1>) -> tensor<84x61x52x56x47x17xi1>
    %2 = tosa.intdiv %arg5, %arg6 : (tensor<27x55x98x30x85xi32>, tensor<1x55x1x1x1xi32>) -> tensor<27x55x98x30x85xi32>
    %3 = tosa.ceil %0 : (tensor<42x246x179x70xf32>) -> tensor<42x246x179x70xf32>
    %4 = tosa.ceil %3 : (tensor<42x246x179x70xf32>) -> tensor<42x246x179x70xf32>
    %5 = tosa.greater %2, %2 : (tensor<27x55x98x30x85xi32>, tensor<27x55x98x30x85xi32>) -> tensor<27x55x98x30x85xi1>
    %t_6 = tosa.const_shape {values = dense<[ 1, 1, 2, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %6 = tosa.tile %4, %t_6 : (tensor<42x246x179x70xf32>, !tosa.shape<4>) -> tensor<42x246x358x140xf32>
    %7 = tosa.logical_xor %5, %5 : (tensor<27x55x98x30x85xi1>, tensor<27x55x98x30x85xi1>) -> tensor<27x55x98x30x85xi1>
    %8 = tosa.reduce_sum %4 {axis = 1 : i32} : (tensor<42x246x179x70xf32>) -> tensor<42x1x179x70xf32>
    %9 = tosa.reduce_all %arg7 {axis = 3 : i32} : (tensor<24x59x61x2xi1>) -> tensor<24x59x61x1xi1>
    %10 = tosa.reduce_sum %9 {axis = 2 : i32} : (tensor<24x59x61x1xi1>) -> tensor<24x59x1x1xi1>
    return %1, %6, %7, %8, %10 : tensor<84x61x52x56x47x17xi1>, tensor<42x246x358x140xf32>, tensor<27x55x98x30x85xi1>, tensor<42x1x179x70xf32>, tensor<24x59x1x1xi1>
  }
}
