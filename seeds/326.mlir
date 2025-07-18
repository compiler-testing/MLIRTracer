module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<82x39xf32>) -> (tensor<i1>, tensor<82x39xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.reciprocal %arg2 : (tensor<82x39xf32>) -> tensor<82x39xf32>
    return %0, %1 : tensor<i1>, tensor<82x39xf32>
  }
}
