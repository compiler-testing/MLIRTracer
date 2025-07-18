module {
  func.func @main(%arg0: tensor<60x11x32xi32>, %arg1: tensor<35x13xf32>) -> (tensor<35x13xf32>, tensor<60x11x32xi32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<60x11x32xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<60x11x32xi32>
    %1 = tosa.sigmoid %arg1 : (tensor<35x13xf32>) -> tensor<35x13xf32>
    %2 = tosa.maximum %0, %0 : (tensor<60x11x32xi32>, tensor<60x11x32xi32>) -> tensor<60x11x32xi32>
    return %1, %2 : tensor<35x13xf32>, tensor<60x11x32xi32>
  }
}
