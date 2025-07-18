module {
  func.func @main(%arg0: tensor<66x71x78xf32>) -> tensor<66x71x78xf32> {
    %0 = tosa.exp %arg0 : (tensor<66x71x78xf32>) -> tensor<66x71x78xf32>
    %in_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<66x71x78xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<66x71x78xf32>
    return %1 : tensor<66x71x78xf32>
  }
}
