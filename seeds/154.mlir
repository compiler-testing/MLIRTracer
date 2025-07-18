module {
  func.func @main(%arg0: tensor<55x29x96x49x79x38xf32>, %arg1: tensor<i1>) -> (tensor<55x29x96x49x79x38xf32>, tensor<i1>, tensor<55x29x96x49x79x38xi1>) {
    %0 = tosa.floor %arg0 : (tensor<55x29x96x49x79x38xf32>) -> tensor<55x29x96x49x79x38xf32>
    %1 = tosa.ceil %0 : (tensor<55x29x96x49x79x38xf32>) -> tensor<55x29x96x49x79x38xf32>
    %2 = tosa.bitwise_not %arg1 : (tensor<i1>) -> tensor<i1>
    %3 = tosa.greater_equal %1, %1 : (tensor<55x29x96x49x79x38xf32>, tensor<55x29x96x49x79x38xf32>) -> tensor<55x29x96x49x79x38xi1>
    %4 = tosa.logical_not %2 : (tensor<i1>) -> tensor<i1>
    %5 = tosa.log %1 : (tensor<55x29x96x49x79x38xf32>) -> tensor<55x29x96x49x79x38xf32>
    %in_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %6 = tosa.negate %4, %in_zp_6, %out_zp_6 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %7 = tosa.arithmetic_right_shift %3, %3 {round = true} : (tensor<55x29x96x49x79x38xi1>, tensor<55x29x96x49x79x38xi1>) -> tensor<55x29x96x49x79x38xi1>
    return %5, %6, %7 : tensor<55x29x96x49x79x38xf32>, tensor<i1>, tensor<55x29x96x49x79x38xi1>
  }
}
