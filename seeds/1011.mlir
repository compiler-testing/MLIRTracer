module {
  func.func @main(%arg0: tensor<59x91x57x66xi1>) -> tensor<59x91x57x66xi1> {
    %0 = tosa.abs %arg0 : (tensor<59x91x57x66xi1>) -> tensor<59x91x57x66xi1>
    return %0 : tensor<59x91x57x66xi1>
  }
}
