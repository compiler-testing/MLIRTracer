module {
  func.func @main(%arg0: tensor<81xi1>, %arg1: tensor<28x31x71x69x76xf32>) -> (tensor<1xi1>, tensor<28x31x71x69x76xf32>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<81xi1>) -> tensor<1xi1>
    %1 = tosa.bitwise_not %0 : (tensor<1xi1>) -> tensor<1xi1>
    %2 = tosa.tanh %arg1 : (tensor<28x31x71x69x76xf32>) -> tensor<28x31x71x69x76xf32>
    return %1, %2 : tensor<1xi1>, tensor<28x31x71x69x76xf32>
  }
}
