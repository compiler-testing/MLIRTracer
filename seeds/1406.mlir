module {
  func.func @main(%arg0: tensor<33x51x9x69x46xf32>) -> tensor<33x51x9x69x46xf32> {
    %0 = tosa.rsqrt %arg0 : (tensor<33x51x9x69x46xf32>) -> tensor<33x51x9x69x46xf32>
    return %0 : tensor<33x51x9x69x46xf32>
  }
}
