module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<70xf32>, %arg3: tensor<55x14x21x22xf32>, %arg4: tensor<74x26x34x40xf32>, %arg5: tensor<74xf32>, %arg6: tensor<87xi1>) -> (tensor<i1>, tensor<1xi1>, tensor<55x55x57x74xf32>, tensor<i32>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.rsqrt %arg2 : (tensor<70xf32>) -> tensor<70xf32>
    %2 = tosa.bitwise_and %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 55, 55, 57, 74>} : (tensor<55x14x21x22xf32>, tensor<74x26x34x40xf32>, tensor<74xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<55x55x57x74xf32>
    %4 = tosa.argmax %1 {axis = 0 : i32} : (tensor<70xf32>) -> tensor<i32>
    %5 = tosa.reduce_all %arg6 {axis = 0 : i32} : (tensor<87xi1>) -> tensor<1xi1>
    %6 = tosa.exp %3 : (tensor<55x55x57x74xf32>) -> tensor<55x55x57x74xf32>
    %7 = tosa.sub %4, %4 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %2, %5, %6, %7 : tensor<i1>, tensor<1xi1>, tensor<55x55x57x74xf32>, tensor<i32>
  }
}
