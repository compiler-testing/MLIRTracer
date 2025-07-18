module {
  func.func @main(%arg0: tensor<53x52x4xi1>) -> tensor<53x52x4xi1> {
    %0 = tosa.logical_not %arg0 : (tensor<53x52x4xi1>) -> tensor<53x52x4xi1>
    return %0 : tensor<53x52x4xi1>
  }
}
