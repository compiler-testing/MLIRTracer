module {
  func.func @main(%arg0: tensor<34xi32>, %arg1: tensor<34xi32>) -> tensor<34xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<34xi32>, tensor<34xi32>) -> tensor<34xi1>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<34xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<34xi1>
    return %1 : tensor<34xi1>
  }
}
