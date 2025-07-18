module {
  func.func @main(%arg0: tensor<7xi64>, %arg1: tensor<1xi64>, %arg2: tensor<65x1xi1>, %arg3: tensor<71x52x54xf32>, %arg4: tensor<1x52x1xf32>) -> (tensor<1x1xi1>, tensor<71x52x54xf32>, tensor<7xi64>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<7xi64>, tensor<1xi64>) -> tensor<7xi64>
    %1 = tosa.minimum %0, %0 : (tensor<7xi64>, tensor<7xi64>) -> tensor<7xi64>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<7xi64>, tensor<7xi64>) -> tensor<7xi64>
    %3 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<65x1xi1>) -> tensor<1x1xi1>
    %4 = tosa.bitwise_and %2, %2 : (tensor<7xi64>, tensor<7xi64>) -> tensor<7xi64>
    %5 = tosa.bitwise_or %4, %1 : (tensor<7xi64>, tensor<7xi64>) -> tensor<7xi64>
    %6 = tosa.sub %5, %1 : (tensor<7xi64>, tensor<7xi64>) -> tensor<7xi64>
    %7 = tosa.pow %arg3, %arg4 : (tensor<71x52x54xf32>, tensor<1x52x1xf32>) -> tensor<71x52x54xf32>
    %8 = tosa.logical_right_shift %6, %6 : (tensor<7xi64>, tensor<7xi64>) -> tensor<7xi64>
    return %3, %7, %8 : tensor<1x1xi1>, tensor<71x52x54xf32>, tensor<7xi64>
  }
}
