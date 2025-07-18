module {
  func.func @main(%arg0: tensor<29x29x67xi64>, %arg1: tensor<29x1x1xi64>, %arg2: tensor<44x36x24x14x2xf32>) -> (tensor<1x1x29x67xi1>, tensor<44x36x24x14x2xf32>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<29x29x67xi64>, tensor<29x1x1xi64>) -> tensor<29x29x67xi64>
    %1 = tosa.bitwise_not %0 : (tensor<29x29x67xi64>) -> tensor<29x29x67xi64>
    %2 = tosa.identity %1 : (tensor<29x29x67xi64>) -> tensor<29x29x67xi64>
    %r_3 = tosa.const_shape {values = dense<[ 1, 29, 29, 67 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.reshape %2, %r_3 : (tensor<29x29x67xi64>, !tosa.shape<4>) -> tensor<1x29x29x67xi64>
    %4 = tosa.logical_left_shift %3, %3 : (tensor<1x29x29x67xi64>, tensor<1x29x29x67xi64>) -> tensor<1x29x29x67xi64>
    %5 = tosa.add %4, %4 : (tensor<1x29x29x67xi64>, tensor<1x29x29x67xi64>) -> tensor<1x29x29x67xi64>
    %6 = tosa.reduce_min %5 {axis = 1 : i32} : (tensor<1x29x29x67xi64>) -> tensor<1x1x29x67xi64>
    %7 = tosa.equal %6, %6 : (tensor<1x1x29x67xi64>, tensor<1x1x29x67xi64>) -> tensor<1x1x29x67xi1>
    %8 = tosa.sub %7, %7 : (tensor<1x1x29x67xi1>, tensor<1x1x29x67xi1>) -> tensor<1x1x29x67xi1>
    %9 = tosa.bitwise_and %8, %7 : (tensor<1x1x29x67xi1>, tensor<1x1x29x67xi1>) -> tensor<1x1x29x67xi1>
    %10 = tosa.arithmetic_right_shift %9, %8 {round = true} : (tensor<1x1x29x67xi1>, tensor<1x1x29x67xi1>) -> tensor<1x1x29x67xi1>
    %11 = tosa.bitwise_or %10, %7 : (tensor<1x1x29x67xi1>, tensor<1x1x29x67xi1>) -> tensor<1x1x29x67xi1>
    %12 = tosa.ceil %arg2 : (tensor<44x36x24x14x2xf32>) -> tensor<44x36x24x14x2xf32>
    return %11, %12 : tensor<1x1x29x67xi1>, tensor<44x36x24x14x2xf32>
  }
}
