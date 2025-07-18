module {
  func.func @main(%arg0: tensor<63xf32>) -> tensor<63xf32> {
    %0 = tosa.log %arg0 : (tensor<63xf32>) -> tensor<63xf32>
    return %0 : tensor<63xf32>
  }
}
