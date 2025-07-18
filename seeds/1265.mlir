module {
  func.func @main(%arg0: tensor<43x74xi64>, %arg1: tensor<43x1xi64>, %arg2: tensor<86x15xi1>) -> (tensor<43x1xi64>, tensor<43x74xi1>, tensor<1x1xi1>, tensor<1x1xi1>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<43x74xi64>, tensor<43x1xi64>) -> tensor<43x74xi64>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<43x74xi64>) -> tensor<43x1xi64>
    %2 = tosa.reduce_all %arg2 {axis = 1 : i32} : (tensor<86x15xi1>) -> tensor<86x1xi1>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<86x1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<86x1xi1>
    %4 = tosa.reduce_any %3 {axis = 0 : i32} : (tensor<86x1xi1>) -> tensor<1x1xi1>
    %5 = tosa.reduce_all %3 {axis = 1 : i32} : (tensor<86x1xi1>) -> tensor<86x1xi1>
    %6 = tosa.logical_xor %5, %5 : (tensor<86x1xi1>, tensor<86x1xi1>) -> tensor<86x1xi1>
    %7 = tosa.greater_equal %0, %0 : (tensor<43x74xi64>, tensor<43x74xi64>) -> tensor<43x74xi1>
    %8 = tosa.arithmetic_right_shift %4, %4 {round = true} : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %9 = tosa.reduce_product %6 {axis = 0 : i32} : (tensor<86x1xi1>) -> tensor<1x1xi1>
    return %1, %7, %8, %9 : tensor<43x1xi64>, tensor<43x74xi1>, tensor<1x1xi1>, tensor<1x1xi1>
  }
}
