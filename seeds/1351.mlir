module {
  func.func @main(%arg0: tensor<22x89xi8>, %arg1: tensor<1x1xi8>, %arg2: tensor<100x82x12x67xf32>, %arg3: tensor<9x64x85x14xf32>, %arg4: tensor<9xf32>) -> (tensor<100x229x109x9xf32>, tensor<22x89xi8>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<22x89xi8>, tensor<1x1xi8>) -> tensor<22x89xi8>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 100, 229, 109, 9>} : (tensor<100x82x12x67xf32>, tensor<9x64x85x14xf32>, tensor<9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<100x229x109x9xf32>
    %2 = tosa.reverse %0 {axis = 0 : i32} : (tensor<22x89xi8>) -> tensor<22x89xi8>
    return %1, %2 : tensor<100x229x109x9xf32>, tensor<22x89xi8>
  }
}
