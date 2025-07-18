module {
  func.func @main(%arg0: tensor<30x75x29x25xi32>, %arg1: tensor<30x1x1x1xi32>) -> tensor<30x75x29x25xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<30x75x29x25xi32>, tensor<30x1x1x1xi32>) -> tensor<30x75x29x25xi1>
    return %0 : tensor<30x75x29x25xi1>
  }
}
