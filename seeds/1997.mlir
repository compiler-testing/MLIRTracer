module {
  func.func @main(%arg0: tensor<74x84x1x74x34x12xi32>, %arg1: tensor<1x1x1x1x34x1xi32>) -> tensor<74x84x1x74x34x12xi32> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<74x84x1x74x34x12xi32>, tensor<1x1x1x1x34x1xi32>) -> tensor<74x84x1x74x34x12xi32>
    return %0 : tensor<74x84x1x74x34x12xi32>
  }
}
