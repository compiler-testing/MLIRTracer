module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<23x27xf32>) -> (tensor<i1>, tensor<23x27xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<i1>) -> tensor<i1>
    %1 = tosa.tanh %arg1 : (tensor<23x27xf32>) -> tensor<23x27xf32>
    return %0, %1 : tensor<i1>, tensor<23x27xf32>
  }
}
