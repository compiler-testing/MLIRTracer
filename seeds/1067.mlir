module {
  func.func @main(%arg0: tensor<22xi8>, %arg1: tensor<1xi8>, %arg2: tensor<71x20x34x52xf32>, %arg3: tensor<73x23x16x31xf32>, %arg4: tensor<73xf32>) -> (tensor<71x45x53x73xf32>, tensor<22xi8>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<22xi8>, tensor<1xi8>) -> tensor<22xi8>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 71, 45, 53, 73>} : (tensor<71x20x34x52xf32>, tensor<73x23x16x31xf32>, tensor<73xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<71x45x53x73xf32>
    %2 = tosa.bitwise_xor %0, %0 : (tensor<22xi8>, tensor<22xi8>) -> tensor<22xi8>
    %3 = tosa.reverse %2 {axis = 0 : i32} : (tensor<22xi8>) -> tensor<22xi8>
    return %1, %3 : tensor<71x45x53x73xf32>, tensor<22xi8>
  }
}
