module {
  func.func @main(%arg0: tensor<3x36x6xf32>) -> (tensor<3x36x6xi1>, tensor<3x36x6xi1>, tensor<3x36x6xi1>, tensor<3x36x6xf32>) {
    %0 = tosa.floor %arg0 : (tensor<3x36x6xf32>) -> tensor<3x36x6xf32>
    %1 = tosa.greater %0, %0 : (tensor<3x36x6xf32>, tensor<3x36x6xf32>) -> tensor<3x36x6xi1>
    %2 = tosa.equal %0, %0 : (tensor<3x36x6xf32>, tensor<3x36x6xf32>) -> tensor<3x36x6xi1>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<3x36x6xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<3x36x6xi1>
    %4 = tosa.greater_equal %0, %0 : (tensor<3x36x6xf32>, tensor<3x36x6xf32>) -> tensor<3x36x6xi1>
    %5 = tosa.log %0 : (tensor<3x36x6xf32>) -> tensor<3x36x6xf32>
    return %2, %3, %4, %5 : tensor<3x36x6xi1>, tensor<3x36x6xi1>, tensor<3x36x6xi1>, tensor<3x36x6xf32>
  }
}
