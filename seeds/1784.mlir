module {
  func.func @main(%arg0: tensor<i8>) -> tensor<i8> {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<i8>, tensor<1xi8>, tensor<1xi8>) -> tensor<i8>
    return %0 : tensor<i8>
  }
}
