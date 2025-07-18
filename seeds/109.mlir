module {
  func.func @main(%arg0: tensor<47x59x67x32x77x11xi64>, %arg1: tensor<47x59x67x32x77x1xi64>, %arg2: tensor<95x99x40x55x22xf32>) -> (tensor<95x99x40x55x22xf32>, tensor<47x59x67x32x77x11xi64>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<47x59x67x32x77x11xi64>, tensor<47x59x67x32x77x1xi64>) -> tensor<47x59x67x32x77x11xi64>
    %1 = tosa.exp %arg2 : (tensor<95x99x40x55x22xf32>) -> tensor<95x99x40x55x22xf32>
    %2 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<47x59x67x32x77x11xi64>, tensor<47x59x67x32x77x11xi64>) -> tensor<47x59x67x32x77x11xi64>
    return %1, %2 : tensor<95x99x40x55x22xf32>, tensor<47x59x67x32x77x11xi64>
  }
}
