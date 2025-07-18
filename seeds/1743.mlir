module {
  func.func @main(%arg0: tensor<15x69x35xf32>) -> tensor<15x69x35xf32> {
    %0 = tosa.abs %arg0 : (tensor<15x69x35xf32>) -> tensor<15x69x35xf32>
    return %0 : tensor<15x69x35xf32>
  }
}
