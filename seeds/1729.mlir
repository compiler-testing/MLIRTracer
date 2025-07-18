module {
  func.func @main(%arg0: tensor<20x77x82xi1>, %arg1: tensor<20x1x82xi1>, %arg2: tensor<60x7x79x60x4x95xf32>) -> (tensor<20x77x82xi1>, tensor<20x77x82xi1>, tensor<60x7x79x60x4x95xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<20x77x82xi1>, tensor<20x1x82xi1>) -> tensor<20x77x82xi1>
    %1 = tosa.logical_xor %0, %0 : (tensor<20x77x82xi1>, tensor<20x77x82xi1>) -> tensor<20x77x82xi1>
    %2 = tosa.abs %1 : (tensor<20x77x82xi1>) -> tensor<20x77x82xi1>
    %3 = tosa.rsqrt %arg2 : (tensor<60x7x79x60x4x95xf32>) -> tensor<60x7x79x60x4x95xf32>
    %4 = tosa.maximum %3, %3 : (tensor<60x7x79x60x4x95xf32>, tensor<60x7x79x60x4x95xf32>) -> tensor<60x7x79x60x4x95xf32>
    %5 = tosa.identity %4 : (tensor<60x7x79x60x4x95xf32>) -> tensor<60x7x79x60x4x95xf32>
    %6 = tosa.logical_right_shift %1, %1 : (tensor<20x77x82xi1>, tensor<20x77x82xi1>) -> tensor<20x77x82xi1>
    %7 = tosa.sub %5, %5 : (tensor<60x7x79x60x4x95xf32>, tensor<60x7x79x60x4x95xf32>) -> tensor<60x7x79x60x4x95xf32>
    return %2, %6, %7 : tensor<20x77x82xi1>, tensor<20x77x82xi1>, tensor<60x7x79x60x4x95xf32>
  }
}
