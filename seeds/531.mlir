module {
  func.func @main(%arg0: tensor<87x68x11x18x95x12xi32>, %arg1: tensor<87x68x11x1x1x12xi32>, %arg2: tensor<71x29x14x94x20x1xf32>) -> (tensor<87x68x11x18x95x12xi32>, tensor<71x29x14x94x20x1xf32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<87x68x11x18x95x12xi32>, tensor<87x68x11x1x1x12xi32>) -> tensor<87x68x11x18x95x12xi32>
    %1 = tosa.ceil %arg2 : (tensor<71x29x14x94x20x1xf32>) -> tensor<71x29x14x94x20x1xf32>
    return %0, %1 : tensor<87x68x11x18x95x12xi32>, tensor<71x29x14x94x20x1xf32>
  }
}
