module {
  func.func @main(%arg0: tensor<11x64x100xi64>, %arg1: tensor<11x1x1xi64>, %arg2: tensor<97x12x71xi1>) -> (tensor<11x64x100xi1>, tensor<97x12x1xi1>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<11x64x100xi64>, tensor<11x1x1xi64>) -> tensor<11x64x100xi64>
    %1 = tosa.reduce_all %arg2 {axis = 2 : i32} : (tensor<97x12x71xi1>) -> tensor<97x12x1xi1>
    %2 = tosa.identity %0 : (tensor<11x64x100xi64>) -> tensor<11x64x100xi64>
    %3 = tosa.logical_right_shift %2, %2 : (tensor<11x64x100xi64>, tensor<11x64x100xi64>) -> tensor<11x64x100xi64>
    %4 = tosa.equal %3, %0 : (tensor<11x64x100xi64>, tensor<11x64x100xi64>) -> tensor<11x64x100xi1>
    %5 = tosa.bitwise_or %4, %4 : (tensor<11x64x100xi1>, tensor<11x64x100xi1>) -> tensor<11x64x100xi1>
    %6 = tosa.add %1, %1 : (tensor<97x12x1xi1>, tensor<97x12x1xi1>) -> tensor<97x12x1xi1>
    return %5, %6 : tensor<11x64x100xi1>, tensor<97x12x1xi1>
  }
}
