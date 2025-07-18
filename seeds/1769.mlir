module {
  func.func @main(%arg0: tensor<45x69x81x29xi64>) -> tensor<45x69x81x29xi64> {
    %0 = tosa.identity %arg0 : (tensor<45x69x81x29xi64>) -> tensor<45x69x81x29xi64>
    return %0 : tensor<45x69x81x29xi64>
  }
}
