module {
  func.func @main(%arg0: tensor<80x62x4xf32>, %arg1: tensor<80x1x4xf32>, %arg2: tensor<74xi32>, %arg3: tensor<1xi32>, %arg4: tensor<63xi32>, %arg5: tensor<63xi32>) -> (tensor<74xi1>, tensor<80x62x4xi1>, tensor<63xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<80x62x4xf32>, tensor<80x1x4xf32>) -> tensor<80x62x4xi1>
    %1 = tosa.equal %arg2, %arg3 : (tensor<74xi32>, tensor<1xi32>) -> tensor<74xi1>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<74xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<74xi1>
    %3 = tosa.logical_right_shift %0, %0 : (tensor<80x62x4xi1>, tensor<80x62x4xi1>) -> tensor<80x62x4xi1>
    %4 = tosa.intdiv %arg4, %arg5 : (tensor<63xi32>, tensor<63xi32>) -> tensor<63xi32>
    %5 = tosa.bitwise_or %3, %0 : (tensor<80x62x4xi1>, tensor<80x62x4xi1>) -> tensor<80x62x4xi1>
    %6 = tosa.greater_equal %4, %4 : (tensor<63xi32>, tensor<63xi32>) -> tensor<63xi1>
    return %2, %5, %6 : tensor<74xi1>, tensor<80x62x4xi1>, tensor<63xi1>
  }
}
