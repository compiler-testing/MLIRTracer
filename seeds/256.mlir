module {
  func.func @main(%arg0: tensor<48x70x10xi64>, %arg1: tensor<1x70x10xi64>, %arg2: tensor<59x70x66xi1>) -> (tensor<59x1x66xi1>, tensor<48x70xi32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<48x70x10xi64>, tensor<1x70x10xi64>) -> tensor<48x70x10xi64>
    %1 = tosa.argmax %0 {axis = 2 : i32} : (tensor<48x70x10xi64>) -> tensor<48x70xi32>
    %2 = tosa.maximum %1, %1 : (tensor<48x70xi32>, tensor<48x70xi32>) -> tensor<48x70xi32>
    %3 = tosa.reduce_any %arg2 {axis = 1 : i32} : (tensor<59x70x66xi1>) -> tensor<59x1x66xi1>
    %4 = tosa.logical_right_shift %3, %3 : (tensor<59x1x66xi1>, tensor<59x1x66xi1>) -> tensor<59x1x66xi1>
    %5 = tosa.arithmetic_right_shift %2, %2 {round = false} : (tensor<48x70xi32>, tensor<48x70xi32>) -> tensor<48x70xi32>
    return %4, %5 : tensor<59x1x66xi1>, tensor<48x70xi32>
  }
}
