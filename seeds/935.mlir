module {
  func.func @main(%arg0: tensor<42x70x24x34x43xf32>, %arg1: tensor<82x7xi1>, %arg2: tensor<82x7xi1>) -> (tensor<2x10x7x12x1xf32>, tensor<1x4xi1>) {
    %0 = tosa.exp %arg0 : (tensor<42x70x24x34x43xf32>) -> tensor<42x70x24x34x43xf32>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<82x7xi1>, tensor<82x7xi1>) -> tensor<82x7xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 14, 5, 17, 22, 21 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_2_size = tosa.const_shape {values = dense<[ 2, 10, 7, 12, 1 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %2 = tosa.slice %0, %s_2_start, %s_2_size : (tensor<42x70x24x34x43xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<2x10x7x12x1xf32>
    %s_3_start = tosa.const_shape {values = dense<[ 42, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_3_size = tosa.const_shape {values = dense<[ 6, 4 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.slice %1, %s_3_start, %s_3_size : (tensor<82x7xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<6x4xi1>
    %4 = tosa.exp %2 : (tensor<2x10x7x12x1xf32>) -> tensor<2x10x7x12x1xf32>
    %5 = tosa.reduce_all %3 {axis = 0 : i32} : (tensor<6x4xi1>) -> tensor<1x4xi1>
    return %4, %5 : tensor<2x10x7x12x1xf32>, tensor<1x4xi1>
  }
}
