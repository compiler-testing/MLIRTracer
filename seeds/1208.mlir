module {
  func.func @main(%arg0: tensor<43xf32>) -> tensor<43xf32> {
    %0 = tosa.rsqrt %arg0 : (tensor<43xf32>) -> tensor<43xf32>
    %1 = tosa.tanh %0 : (tensor<43xf32>) -> tensor<43xf32>
    %2 = tosa.log %1 : (tensor<43xf32>) -> tensor<43xf32>
    return %2 : tensor<43xf32>
  }
}
