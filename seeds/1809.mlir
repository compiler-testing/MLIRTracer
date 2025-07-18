module {
  func.func @main(%arg0: tensor<26x61x4xf32>, %arg1: tensor<44x59xi1>) -> (tensor<26x61x4xf32>, tensor<44x59xi1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<26x61x4xf32>) -> tensor<26x61x4xf32>
    %1 = tosa.logical_not %arg1 : (tensor<44x59xi1>) -> tensor<44x59xi1>
    return %0, %1 : tensor<26x61x4xf32>, tensor<44x59xi1>
  }
}
