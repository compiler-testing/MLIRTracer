module {
  func.func @main(%arg0: tensor<28xf32>) -> tensor<28xf32> {
    %0 = tosa.rsqrt %arg0 : (tensor<28xf32>) -> tensor<28xf32>
    %1 = tosa.log %0 : (tensor<28xf32>) -> tensor<28xf32>
    return %1 : tensor<28xf32>
  }
}
