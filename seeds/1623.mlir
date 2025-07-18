module {
  func.func @main(%arg0: tensor<40x2x12x13x84xf32>, %arg1: tensor<74xi64>, %arg2: tensor<90x52x96x78xf32>, %arg3: tensor<72x3x44x49xf32>, %arg4: tensor<72xf32>) -> (tensor<40x2x12x13x84xf32>, tensor<74xi64>, tensor<90x57x236x72xf32>) {
    %0 = tosa.exp %arg0 : (tensor<40x2x12x13x84xf32>) -> tensor<40x2x12x13x84xf32>
    %1 = tosa.reverse %arg1 {axis = 0 : i32} : (tensor<74xi64>) -> tensor<74xi64>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 1>, stride = array<i64: 1, 2>, out_shape = array<i64: 90, 57, 236, 72>} : (tensor<90x52x96x78xf32>, tensor<72x3x44x49xf32>, tensor<72xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<90x57x236x72xf32>
    %3 = tosa.log %2 : (tensor<90x57x236x72xf32>) -> tensor<90x57x236x72xf32>
    return %0, %1, %3 : tensor<40x2x12x13x84xf32>, tensor<74xi64>, tensor<90x57x236x72xf32>
  }
}
