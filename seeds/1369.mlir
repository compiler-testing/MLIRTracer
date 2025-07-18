module {
  func.func @main(%arg0: tensor<35x2x96x1x28x49xf32>) -> tensor<35x2x96x1x28x49xf32> {
    %0 = tosa.sigmoid %arg0 : (tensor<35x2x96x1x28x49xf32>) -> tensor<35x2x96x1x28x49xf32>
    return %0 : tensor<35x2x96x1x28x49xf32>
  }
}
