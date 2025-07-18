module {
  func.func @main(%arg0: tensor<100x6x41x55xf32>, %arg1: tensor<44x56x19x51xf32>, %arg2: tensor<44xf32>, %arg3: tensor<i1>, %arg4: tensor<i1>) -> (tensor<i1>, tensor<100x64x101x44xf32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<100x6x41x55xf32>) -> tensor<100x6x41x55xf32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %0, %arg1, %arg2, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 1>, stride = array<i64: 1, 2>, out_shape = array<i64: 100, 64, 101, 44>} : (tensor<100x6x41x55xf32>, tensor<44x56x19x51xf32>, tensor<44xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<100x64x101x44xf32>
    %2 = tosa.logical_xor %arg3, %arg4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.reverse %1 {axis = 1 : i32} : (tensor<100x64x101x44xf32>) -> tensor<100x64x101x44xf32>
    return %2, %3 : tensor<i1>, tensor<100x64x101x44xf32>
  }
}
