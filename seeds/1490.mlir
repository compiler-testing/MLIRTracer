module {
  func.func @main(%arg0: tensor<i16>) -> tensor<i16> {
    %0 = tosa.clz %arg0 : (tensor<i16>) -> tensor<i16>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<i16>, tensor<1xi16>, tensor<1xi16>) -> tensor<i16>
    return %1 : tensor<i16>
  }
}
