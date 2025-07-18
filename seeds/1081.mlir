module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<9x53x35x93x71x27xf32>, %arg2: tensor<23x91x1xf32>) -> (tensor<9x53x35x93x71x27xf32>, tensor<23x91x1xf32>, tensor<i1>) {
    %0 = tosa.logical_not %arg0 : (tensor<i1>) -> tensor<i1>
    %1 = tosa.log %arg1 : (tensor<9x53x35x93x71x27xf32>) -> tensor<9x53x35x93x71x27xf32>
    %2 = tosa.logical_right_shift %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.reduce_sum %arg2 {axis = 2 : i32} : (tensor<23x91x1xf32>) -> tensor<23x91x1xf32>
    %4 = tosa.logical_xor %2, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %1, %3, %4 : tensor<9x53x35x93x71x27xf32>, tensor<23x91x1xf32>, tensor<i1>
  }
}
