module {
  func.func @main(%arg0: tensor<9xf32>, %arg1: tensor<1xf32>, %arg2: tensor<26x11x53xi16>, %arg3: tensor<1x11x53xi16>) -> (tensor<9xf32>, tensor<26x11x53xi16>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<9xf32>, tensor<1xf32>) -> tensor<9xf32>
    %in_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<9xf32>
    %2 = tosa.bitwise_or %arg2, %arg3 : (tensor<26x11x53xi16>, tensor<1x11x53xi16>) -> tensor<26x11x53xi16>
    return %1, %2 : tensor<9xf32>, tensor<26x11x53xi16>
  }
}
