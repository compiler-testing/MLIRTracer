module {
  func.func @main(%arg0: tensor<10x72x89xi1>) -> tensor<10x72x89xi1> {
    %0 = tosa.clz %arg0 : (tensor<10x72x89xi1>) -> tensor<10x72x89xi1>
    %1 = tosa.abs %0 : (tensor<10x72x89xi1>) -> tensor<10x72x89xi1>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<10x72x89xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<10x72x89xi1>
    return %2 : tensor<10x72x89xi1>
  }
}
