module {
  func.func @main(%arg0: tensor<14x88x45x67xi64>, %arg1: tensor<1x1x45x67xi64>, %arg2: tensor<75x38x82x65x1xf32>, %arg3: tensor<1x1x1x65x1xf32>) -> (tensor<75x38x82x65x1xf32>, tensor<14x88x45x67xi1>, tensor<14x88x1x67xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<14x88x45x67xi64>, tensor<1x1x45x67xi64>) -> tensor<14x88x45x67xi1>
    %1 = tosa.identity %0 : (tensor<14x88x45x67xi1>) -> tensor<14x88x45x67xi1>
    %2 = tosa.pow %arg2, %arg3 : (tensor<75x38x82x65x1xf32>, tensor<1x1x1x65x1xf32>) -> tensor<75x38x82x65x1xf32>
    %3 = tosa.maximum %2, %2 : (tensor<75x38x82x65x1xf32>, tensor<75x38x82x65x1xf32>) -> tensor<75x38x82x65x1xf32>
    %4 = tosa.reverse %0 {axis = 0 : i32} : (tensor<14x88x45x67xi1>) -> tensor<14x88x45x67xi1>
    %5 = tosa.arithmetic_right_shift %4, %0 {round = false} : (tensor<14x88x45x67xi1>, tensor<14x88x45x67xi1>) -> tensor<14x88x45x67xi1>
    %6 = tosa.reduce_min %1 {axis = 2 : i32} : (tensor<14x88x45x67xi1>) -> tensor<14x88x1x67xi1>
    return %3, %5, %6 : tensor<75x38x82x65x1xf32>, tensor<14x88x45x67xi1>, tensor<14x88x1x67xi1>
  }
}
