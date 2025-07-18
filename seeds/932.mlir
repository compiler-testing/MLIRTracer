module {
  func.func @main(%arg0: tensor<48x23xi32>, %arg1: tensor<48x23xi32>, %arg2: tensor<74x36x72xf32>) -> (tensor<48x23xi32>, tensor<74x36x72xf32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<48x23xi32>, tensor<48x23xi32>) -> tensor<48x23xi32>
    %1 = tosa.identity %0 : (tensor<48x23xi32>) -> tensor<48x23xi32>
    %2 = tosa.tanh %arg2 : (tensor<74x36x72xf32>) -> tensor<74x36x72xf32>
    return %1, %2 : tensor<48x23xi32>, tensor<74x36x72xf32>
  }
}
