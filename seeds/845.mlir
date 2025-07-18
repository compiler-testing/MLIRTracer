module {
  func.func @main(%arg0: tensor<25x99x26x59x8xi1>, %arg1: tensor<26x4x40x63xf32>, %arg2: tensor<65x6x47x26xf32>, %arg3: tensor<65xf32>) -> (tensor<25x99x26x59x8xi1>, tensor<26x13x90x65xf32>) {
    %0 = tosa.clz %arg0 : (tensor<25x99x26x59x8xi1>) -> tensor<25x99x26x59x8xi1>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 26, 13, 90, 65>} : (tensor<26x4x40x63xf32>, tensor<65x6x47x26xf32>, tensor<65xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<26x13x90x65xf32>
    %2 = tosa.identity %1 : (tensor<26x13x90x65xf32>) -> tensor<26x13x90x65xf32>
    return %0, %2 : tensor<25x99x26x59x8xi1>, tensor<26x13x90x65xf32>
  }
}
