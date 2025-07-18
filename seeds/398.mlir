module {
  func.func @main(%arg0: tensor<91x10xf32>, %arg1: tensor<68x2x44x98xi1>, %arg2: tensor<1x2x1x1xi1>) -> (tensor<68x2x44x98xi1>, tensor<1x10xf32>) {
    %0 = tosa.ceil %arg0 : (tensor<91x10xf32>) -> tensor<91x10xf32>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<91x10xf32>) -> tensor<1x10xf32>
    %2 = tosa.logical_right_shift %arg1, %arg2 : (tensor<68x2x44x98xi1>, tensor<1x2x1x1xi1>) -> tensor<68x2x44x98xi1>
    %3 = tosa.pow %1, %1 : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %2, %3 : tensor<68x2x44x98xi1>, tensor<1x10xf32>
  }
}
