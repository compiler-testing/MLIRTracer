module {
  func.func @main(%arg0: tensor<82x14x58x63xf32>) -> tensor<82x14x58x63xf32> {
    %0 = tosa.reciprocal %arg0 : (tensor<82x14x58x63xf32>) -> tensor<82x14x58x63xf32>
    return %0 : tensor<82x14x58x63xf32>
  }
}
