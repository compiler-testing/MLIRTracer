module {
  func.func @main(%arg0: tensor<88x88x21x88xi8>, %arg1: tensor<15x56xi1>, %arg2: tensor<15x56xi1>) -> (tensor<15x56xi1>, tensor<88x88x21x88xi8>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<88x88x21x88xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<88x88x21x88xi8>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<15x56xi1>, tensor<15x56xi1>) -> tensor<15x56xi1>
    %2 = tosa.maximum %0, %0 : (tensor<88x88x21x88xi8>, tensor<88x88x21x88xi8>) -> tensor<88x88x21x88xi8>
    return %1, %2 : tensor<15x56xi1>, tensor<88x88x21x88xi8>
  }
}
