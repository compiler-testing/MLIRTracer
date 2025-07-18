module {
  func.func @main(%arg0: tensor<38x53x60xf32>) -> tensor<38x53x60xf32> {
    %in_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<38x53x60xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<38x53x60xf32>
    return %0 : tensor<38x53x60xf32>
  }
}
