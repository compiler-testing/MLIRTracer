module {
  func.func @main(%arg0: tensor<97x65x80x28x53x79xi32>) -> tensor<97x65x80x28x53x79xi32> {
    %0 = tosa.identity %arg0 : (tensor<97x65x80x28x53x79xi32>) -> tensor<97x65x80x28x53x79xi32>
    return %0 : tensor<97x65x80x28x53x79xi32>
  }
}
