module {
  func.func @main(%arg0: tensor<26x84x78x67xf32>) -> tensor<26x84x78x67xf32> {
    %0 = tosa.exp %arg0 : (tensor<26x84x78x67xf32>) -> tensor<26x84x78x67xf32>
    return %0 : tensor<26x84x78x67xf32>
  }
}
