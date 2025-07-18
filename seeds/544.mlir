module {
  func.func @main(%arg0: tensor<66x16x87xf32>, %arg1: tensor<1x16x87xf32>) -> tensor<66x16x87xf32> {
    %0 = tosa.pow %arg0, %arg1 : (tensor<66x16x87xf32>, tensor<1x16x87xf32>) -> tensor<66x16x87xf32>
    %1 = tosa.rsqrt %0 : (tensor<66x16x87xf32>) -> tensor<66x16x87xf32>
    return %1 : tensor<66x16x87xf32>
  }
}
