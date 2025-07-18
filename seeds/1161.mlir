module {
  func.func @main(%arg0: tensor<44x45x32x9xi32>, %arg1: tensor<1x45x1x9xi32>) -> tensor<44x90x32x9xi1> {
    %0 = tosa.greater %arg0, %arg1 : (tensor<44x45x32x9xi32>, tensor<1x45x1x9xi32>) -> tensor<44x45x32x9xi1>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<44x45x32x9xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<44x45x32x9xi1>
    %2 = tosa.bitwise_or %1, %0 : (tensor<44x45x32x9xi1>, tensor<44x45x32x9xi1>) -> tensor<44x45x32x9xi1>
    %3 = tosa.logical_left_shift %2, %1 : (tensor<44x45x32x9xi1>, tensor<44x45x32x9xi1>) -> tensor<44x45x32x9xi1>
    %4 = tosa.concat %3, %0 {axis = 1 : i32} : (tensor<44x45x32x9xi1>, tensor<44x45x32x9xi1>) -> tensor<44x90x32x9xi1>
    return %4 : tensor<44x90x32x9xi1>
  }
}
