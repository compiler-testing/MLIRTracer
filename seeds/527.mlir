module {
  func.func @main(%arg0: tensor<52x94x73x30x84x41xi64>, %arg1: tensor<1x94x73x30x84x1xi64>, %arg2: tensor<90x21xi1>, %arg3: tensor<90x21xi1>) -> (tensor<52x94x73x30x84x41xi64>, tensor<1x21xi1>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<52x94x73x30x84x41xi64>, tensor<1x94x73x30x84x1xi64>) -> tensor<52x94x73x30x84x41xi64>
    %1 = tosa.logical_or %arg2, %arg3 : (tensor<90x21xi1>, tensor<90x21xi1>) -> tensor<90x21xi1>
    %2 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<90x21xi1>, tensor<90x21xi1>) -> tensor<90x21xi1>
    %3 = tosa.reduce_any %2 {axis = 0 : i32} : (tensor<90x21xi1>) -> tensor<1x21xi1>
    return %0, %3 : tensor<52x94x73x30x84x41xi64>, tensor<1x21xi1>
  }
}
