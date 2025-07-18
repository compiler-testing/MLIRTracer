module {
  func.func @main(%arg0: tensor<59x61x97x44x19x45xi1>) -> tensor<59x61x97x44x19x45xi1> {
    %0 = tosa.logical_not %arg0 : (tensor<59x61x97x44x19x45xi1>) -> tensor<59x61x97x44x19x45xi1>
    return %0 : tensor<59x61x97x44x19x45xi1>
  }
}
