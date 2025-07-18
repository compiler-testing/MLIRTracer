module {
  func.func @main(%arg0: tensor<8x54x39x55xf32>, %arg1: tensor<1x1x39x55xf32>, %arg2: tensor<40x21xi1>, %arg3: tensor<1x21xi1>) -> (tensor<40x21xi1>, tensor<8x54x39x55xf32>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<8x54x39x55xf32>, tensor<1x1x39x55xf32>) -> tensor<8x54x39x55xf32>
    %1 = tosa.minimum %0, %0 : (tensor<8x54x39x55xf32>, tensor<8x54x39x55xf32>) -> tensor<8x54x39x55xf32>
    %in_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<8x54x39x55xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<8x54x39x55xf32>
    %3 = tosa.tanh %2 : (tensor<8x54x39x55xf32>) -> tensor<8x54x39x55xf32>
    %in_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.negate %3, %in_zp_4, %out_zp_4 : (tensor<8x54x39x55xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<8x54x39x55xf32>
    %5 = tosa.logical_right_shift %arg2, %arg3 : (tensor<40x21xi1>, tensor<1x21xi1>) -> tensor<40x21xi1>
    %6 = tosa.log %4 : (tensor<8x54x39x55xf32>) -> tensor<8x54x39x55xf32>
    %7 = tosa.tanh %6 : (tensor<8x54x39x55xf32>) -> tensor<8x54x39x55xf32>
    return %5, %7 : tensor<40x21xi1>, tensor<8x54x39x55xf32>
  }
}
