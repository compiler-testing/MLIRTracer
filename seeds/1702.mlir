module {
  func.func @main(%arg0: tensor<60x12x54x28xf32>, %arg1: tensor<41x58x39x78xf32>, %arg2: tensor<41xf32>, %arg3: tensor<72x38x79x92xi32>) -> (tensor<72x38x79x92xi32>, tensor<60x84x94x41xf32>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 60, 84, 94, 41>} : (tensor<60x12x54x28xf32>, tensor<41x58x39x78xf32>, tensor<41xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<60x84x94x41xf32>
    %1 = tosa.bitwise_not %arg3 : (tensor<72x38x79x92xi32>) -> tensor<72x38x79x92xi32>
    %2 = tosa.exp %0 : (tensor<60x84x94x41xf32>) -> tensor<60x84x94x41xf32>
    return %1, %2 : tensor<72x38x79x92xi32>, tensor<60x84x94x41xf32>
  }
}
