module {
  func.func @main(%arg0: tensor<82xi1>, %arg1: tensor<82xi1>, %arg2: tensor<9xf32>) -> (tensor<82xi1>, tensor<9xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<82xi1>, tensor<82xi1>) -> tensor<82xi1>
    %1 = tosa.ceil %arg2 : (tensor<9xf32>) -> tensor<9xf32>
    %2 = tosa.ceil %1 : (tensor<9xf32>) -> tensor<9xf32>
    return %0, %2 : tensor<82xi1>, tensor<9xf32>
  }
}
