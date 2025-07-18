module {
  func.func @main(%arg0: tensor<23xf32>) -> tensor<23xf32> {
    %0 = tosa.rsqrt %arg0 : (tensor<23xf32>) -> tensor<23xf32>
    return %0 : tensor<23xf32>
  }
}
