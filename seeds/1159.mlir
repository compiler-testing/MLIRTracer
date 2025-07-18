module {
  func.func @main(%arg0: tensor<26x63xf32>) -> tensor<26x63xf32> {
    %0 = tosa.sigmoid %arg0 : (tensor<26x63xf32>) -> tensor<26x63xf32>
    return %0 : tensor<26x63xf32>
  }
}
