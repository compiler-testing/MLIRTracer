module {
  func.func @main(%arg0: tensor<25x33x45x14x17x72xi32>, %arg1: tensor<1x33x45x1x17x72xi32>) -> tensor<25x33x45x14x17x72xi32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<25x33x45x14x17x72xi32>, tensor<1x33x45x1x17x72xi32>) -> tensor<25x33x45x14x17x72xi32>
    return %0 : tensor<25x33x45x14x17x72xi32>
  }
}
