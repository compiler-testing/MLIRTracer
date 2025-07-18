module {
  func.func @main(%arg0: tensor<36x54xi16>, %arg1: tensor<31xi64>, %arg2: tensor<31xi64>) -> (tensor<31xi1>, tensor<36x54xi16>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<36x54xi16>) -> tensor<36x54xi16>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<36x54xi16>, tensor<36x54xi16>) -> tensor<36x54xi16>
    %2 = tosa.logical_right_shift %1, %0 : (tensor<36x54xi16>, tensor<36x54xi16>) -> tensor<36x54xi16>
    %3 = tosa.greater_equal %arg1, %arg2 : (tensor<31xi64>, tensor<31xi64>) -> tensor<31xi1>
    %4 = tosa.logical_right_shift %2, %0 : (tensor<36x54xi16>, tensor<36x54xi16>) -> tensor<36x54xi16>
    return %3, %4 : tensor<31xi1>, tensor<36x54xi16>
  }
}
