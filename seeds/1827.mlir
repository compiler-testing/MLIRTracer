module {
  func.func @main(%arg0: tensor<11x99x16x42x83xf32>) -> tensor<11x99x16x42x83xf32> {
    %0 = tosa.rsqrt %arg0 : (tensor<11x99x16x42x83xf32>) -> tensor<11x99x16x42x83xf32>
    %1 = tosa.tanh %0 : (tensor<11x99x16x42x83xf32>) -> tensor<11x99x16x42x83xf32>
    %2 = tosa.rsqrt %1 : (tensor<11x99x16x42x83xf32>) -> tensor<11x99x16x42x83xf32>
    return %2 : tensor<11x99x16x42x83xf32>
  }
}
