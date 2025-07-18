module {
  func.func @main(%arg0: tensor<88x55x31x64x55x62xi8>) -> tensor<88x55x31x64x55x62xi8> {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<88x55x31x64x55x62xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<88x55x31x64x55x62xi8>
    return %0 : tensor<88x55x31x64x55x62xi8>
  }
}
