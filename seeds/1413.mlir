module {
  func.func @main(%arg0: tensor<70x83x70x84x59xf32>, %arg1: tensor<70x1x1x1x1xf32>) -> tensor<70x83x70x84x59xf32> {
    %0 = tosa.pow %arg0, %arg1 : (tensor<70x83x70x84x59xf32>, tensor<70x1x1x1x1xf32>) -> tensor<70x83x70x84x59xf32>
    return %0 : tensor<70x83x70x84x59xf32>
  }
}
