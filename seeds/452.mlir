module {
  func.func @main(%arg0: tensor<43x53x45x36xi32>, %arg1: tensor<43x1x45x36xi32>) -> tensor<43x53x45x36xi32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<43x53x45x36xi32>, tensor<43x1x45x36xi32>) -> tensor<43x53x45x36xi32>
    return %0 : tensor<43x53x45x36xi32>
  }
}
