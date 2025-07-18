module {
  func.func @main(%arg0: tensor<68x23x11x11x28xi32>, %arg1: tensor<1x23x1x1x1xi32>) -> tensor<68x23x11x11x28xi32> {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<68x23x11x11x28xi32>, tensor<1x23x1x1x1xi32>) -> tensor<68x23x11x11x28xi32>
    return %0 : tensor<68x23x11x11x28xi32>
  }
}
