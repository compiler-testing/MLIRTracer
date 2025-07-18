module {
  func.func @main(%arg0: tensor<88x5x77xi1>, %arg1: tensor<1x1x1xi1>) -> tensor<88x5x77xi1> {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<88x5x77xi1>, tensor<1x1x1xi1>) -> tensor<88x5x77xi1>
    %1 = tosa.logical_not %0 : (tensor<88x5x77xi1>) -> tensor<88x5x77xi1>
    return %1 : tensor<88x5x77xi1>
  }
}
