module {
  func.func @main(%arg0: tensor<81x94x33x62x14x10xi64>) -> tensor<81x94x33x62x14x10xi64> {
    %0 = tosa.abs %arg0 : (tensor<81x94x33x62x14x10xi64>) -> tensor<81x94x33x62x14x10xi64>
    return %0 : tensor<81x94x33x62x14x10xi64>
  }
}
