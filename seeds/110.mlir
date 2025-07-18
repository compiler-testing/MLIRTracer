module {
  func.func @main(%arg0: tensor<91x22x26x43xf32>) -> tensor<91x22x26x43xf32> {
    %0 = tosa.log %arg0 : (tensor<91x22x26x43xf32>) -> tensor<91x22x26x43xf32>
    return %0 : tensor<91x22x26x43xf32>
  }
}
