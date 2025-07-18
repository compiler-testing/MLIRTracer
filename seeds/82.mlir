module {
  func.func @main(%arg0: tensor<9x24xf32>, %arg1: tensor<9x24xf32>) -> tensor<9x24xf32> {
    %0 = tosa.pow %arg0, %arg1 : (tensor<9x24xf32>, tensor<9x24xf32>) -> tensor<9x24xf32>
    %1 = tosa.reciprocal %0 : (tensor<9x24xf32>) -> tensor<9x24xf32>
    %in_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<9x24xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<9x24xf32>
    return %2 : tensor<9x24xf32>
  }
}
