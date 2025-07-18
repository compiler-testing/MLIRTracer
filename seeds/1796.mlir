module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<63x91x28xf32>, %arg2: tensor<63x1x1xf32>) -> (tensor<f32>, tensor<63x91x28xf32>) {
    %0 = tosa.tanh %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.minimum %arg1, %arg2 : (tensor<63x91x28xf32>, tensor<63x1x1xf32>) -> tensor<63x91x28xf32>
    return %0, %1 : tensor<f32>, tensor<63x91x28xf32>
  }
}
