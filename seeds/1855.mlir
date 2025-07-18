module {
  func.func @main(%arg0: tensor<99x17x3xi64>, %arg1: tensor<23x80x56xi1>) -> (tensor<1x80x56xi1>, tensor<1x17x1xi1>, tensor<1x17x3xi1>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<99x17x3xi64>) -> tensor<1x17x3xi64>
    %1 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<23x80x56xi1>) -> tensor<1x80x56xi1>
    %2 = tosa.greater_equal %0, %0 : (tensor<1x17x3xi64>, tensor<1x17x3xi64>) -> tensor<1x17x3xi1>
    %3 = tosa.reduce_any %2 {axis = 2 : i32} : (tensor<1x17x3xi1>) -> tensor<1x17x1xi1>
    %4 = tosa.logical_right_shift %3, %3 : (tensor<1x17x1xi1>, tensor<1x17x1xi1>) -> tensor<1x17x1xi1>
    %5 = tosa.greater_equal %0, %0 : (tensor<1x17x3xi64>, tensor<1x17x3xi64>) -> tensor<1x17x3xi1>
    return %1, %4, %5 : tensor<1x80x56xi1>, tensor<1x17x1xi1>, tensor<1x17x3xi1>
  }
}
