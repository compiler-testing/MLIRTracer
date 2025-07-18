module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<41x53x75x73xf32>, %arg2: tensor<57x95x37x100xf32>, %arg3: tensor<57xf32>, %arg4: tensor<3x77x32x81x95xi64>, %arg5: tensor<1x77x1x81x1xi64>, %arg6: tensor<4x22x26x66xi1>, %arg7: tensor<1x22x26x66xi1>) -> (tensor<41x202x187x57xf32>, tensor<3x77x32x81x95xi64>, tensor<f32>, tensor<4x22x26x66xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.floor %0 : (tensor<f32>) -> tensor<f32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 41, 202, 187, 57>} : (tensor<41x53x75x73xf32>, tensor<57x95x37x100xf32>, tensor<57xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<41x202x187x57xf32>
    %3 = tosa.bitwise_xor %arg4, %arg5 : (tensor<3x77x32x81x95xi64>, tensor<1x77x1x81x1xi64>) -> tensor<3x77x32x81x95xi64>
    %4 = tosa.ceil %1 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.logical_or %arg6, %arg7 : (tensor<4x22x26x66xi1>, tensor<1x22x26x66xi1>) -> tensor<4x22x26x66xi1>
    %6 = tosa.bitwise_xor %5, %5 : (tensor<4x22x26x66xi1>, tensor<4x22x26x66xi1>) -> tensor<4x22x26x66xi1>
    return %2, %3, %4, %6 : tensor<41x202x187x57xf32>, tensor<3x77x32x81x95xi64>, tensor<f32>, tensor<4x22x26x66xi1>
  }
}
