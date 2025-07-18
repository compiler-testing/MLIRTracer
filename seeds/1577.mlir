module {
  func.func @main(%arg0: tensor<51x88x20x46xi32>, %arg1: tensor<51x88x20x1xi32>) -> tensor<51x88x20x46xi32> {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<51x88x20x46xi32>, tensor<51x88x20x1xi32>) -> tensor<51x88x20x46xi32>
    return %0 : tensor<51x88x20x46xi32>
  }
}
