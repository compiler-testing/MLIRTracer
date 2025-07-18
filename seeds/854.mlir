module {
  func.func @main(%arg0: tensor<88x3x99x81xi1>) -> tensor<88x3x99x81xi1> {
    %0 = tosa.logical_not %arg0 : (tensor<88x3x99x81xi1>) -> tensor<88x3x99x81xi1>
    return %0 : tensor<88x3x99x81xi1>
  }
}
