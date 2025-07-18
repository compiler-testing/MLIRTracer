module {
  func.func @main(%arg0: tensor<81x27x8xi64>, %arg1: tensor<1x27x8xi64>, %arg2: tensor<32xi1>, %arg3: tensor<1xi1>) -> (tensor<81x27x8xi64>, tensor<32xi1>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<81x27x8xi64>, tensor<1x27x8xi64>) -> tensor<81x27x8xi64>
    %1 = tosa.logical_and %arg2, %arg3 : (tensor<32xi1>, tensor<1xi1>) -> tensor<32xi1>
    return %0, %1 : tensor<81x27x8xi64>, tensor<32xi1>
  }
}
