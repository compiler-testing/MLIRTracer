module {
  func.func @main(%arg0: tensor<66x9x88x10xi8>, %arg1: tensor<1x1x88x10xi8>) -> tensor<66x9x88x10xi1> {
    %0 = tosa.greater %arg0, %arg1 : (tensor<66x9x88x10xi8>, tensor<1x1x88x10xi8>) -> tensor<66x9x88x10xi1>
    return %0 : tensor<66x9x88x10xi1>
  }
}
