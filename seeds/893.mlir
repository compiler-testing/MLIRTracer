module {
  func.func @main(%arg0: tensor<23x77xf32>, %arg1: tensor<7x77x98x67xi1>) -> (tensor<23x77xf32>, tensor<7x154x1x67xi1>) {
    %0 = tosa.log %arg0 : (tensor<23x77xf32>) -> tensor<23x77xf32>
    %1 = tosa.ceil %0 : (tensor<23x77xf32>) -> tensor<23x77xf32>
    %2 = tosa.logical_not %arg1 : (tensor<7x77x98x67xi1>) -> tensor<7x77x98x67xi1>
    %3 = tosa.concat %2, %2 {axis = 1 : i32} : (tensor<7x77x98x67xi1>, tensor<7x77x98x67xi1>) -> tensor<7x154x98x67xi1>
    %4 = tosa.logical_and %3, %3 : (tensor<7x154x98x67xi1>, tensor<7x154x98x67xi1>) -> tensor<7x154x98x67xi1>
    %5 = tosa.logical_right_shift %3, %4 : (tensor<7x154x98x67xi1>, tensor<7x154x98x67xi1>) -> tensor<7x154x98x67xi1>
    %6 = tosa.pow %1, %1 : (tensor<23x77xf32>, tensor<23x77xf32>) -> tensor<23x77xf32>
    %7 = tosa.reduce_product %5 {axis = 2 : i32} : (tensor<7x154x98x67xi1>) -> tensor<7x154x1x67xi1>
    return %6, %7 : tensor<23x77xf32>, tensor<7x154x1x67xi1>
  }
}
