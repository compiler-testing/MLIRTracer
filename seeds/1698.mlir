module {
  func.func @main(%arg0: tensor<26x44x1x10x90xi1>, %arg1: tensor<26x44x1x1x90xi1>, %arg2: tensor<76x45x71x31x91xf32>) -> (tensor<26x44x1x10x90xi1>, tensor<76x45x71x31x91xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<26x44x1x10x90xi1>, tensor<26x44x1x1x90xi1>) -> tensor<26x44x1x10x90xi1>
    %1 = tosa.abs %0 : (tensor<26x44x1x10x90xi1>) -> tensor<26x44x1x10x90xi1>
    %2 = tosa.reciprocal %arg2 : (tensor<76x45x71x31x91xf32>) -> tensor<76x45x71x31x91xf32>
    return %1, %2 : tensor<26x44x1x10x90xi1>, tensor<76x45x71x31x91xf32>
  }
}
