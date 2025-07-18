module {
  func.func @main(%arg0: tensor<97x89xi1>, %arg1: tensor<44x81x7xf32>) -> (tensor<97x1xi1>, tensor<44x81x7xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<97x89xi1>) -> tensor<97x1xi1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<97x1xi1>, tensor<97x1xi1>) -> tensor<97x1xi1>
    %2 = tosa.exp %arg1 : (tensor<44x81x7xf32>) -> tensor<44x81x7xf32>
    %3 = tosa.greater_equal %2, %2 : (tensor<44x81x7xf32>, tensor<44x81x7xf32>) -> tensor<44x81x7xi1>
    return %1, %3 : tensor<97x1xi1>, tensor<44x81x7xi1>
  }
}
