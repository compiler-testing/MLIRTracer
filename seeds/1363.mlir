module {
  func.func @main(%arg0: tensor<12x54x5x44xf32>, %arg1: tensor<i1>) -> (tensor<i1>, tensor<12x54x5x44xf32>, tensor<10x10x6x16xi1>) {
    %0 = tosa.floor %arg0 : (tensor<12x54x5x44xf32>) -> tensor<12x54x5x44xf32>
    %1 = tosa.logical_not %arg1 : (tensor<i1>) -> tensor<i1>
    %2 = tosa.equal %0, %0 : (tensor<12x54x5x44xf32>, tensor<12x54x5x44xf32>) -> tensor<12x54x5x44xi1>
    %3 = tosa.logical_and %2, %2 : (tensor<12x54x5x44xi1>, tensor<12x54x5x44xi1>) -> tensor<12x54x5x44xi1>
    %4 = tosa.reduce_product %3 {axis = 2 : i32} : (tensor<12x54x5x44xi1>) -> tensor<12x54x1x44xi1>
    %5 = tosa.concat %4, %4 {axis = 0 : i32} : (tensor<12x54x1x44xi1>, tensor<12x54x1x44xi1>) -> tensor<24x54x1x44xi1>
    %6 = tosa.reduce_sum %5 {axis = 3 : i32} : (tensor<24x54x1x44xi1>) -> tensor<24x54x1x1xi1>
    %7 = tosa.maximum %0, %0 : (tensor<12x54x5x44xf32>, tensor<12x54x5x44xf32>) -> tensor<12x54x5x44xf32>
    %t_8 = tosa.const_shape {values = dense<[ 2, 3, 1, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %8 = tosa.tile %6, %t_8 : (tensor<24x54x1x1xi1>, !tosa.shape<4>) -> tensor<48x162x1x2xi1>
    %9 = tosa.bitwise_not %8 : (tensor<48x162x1x2xi1>) -> tensor<48x162x1x2xi1>
    %s_10_start = tosa.const_shape {values = dense<[ 29, 26, 0, 0 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_10_size = tosa.const_shape {values = dense<[ 5, 10, 6, 8 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.slice %9, %s_10_start, %s_10_size : (tensor<48x162x1x2xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x10x6x8xi1>
    %t_11 = tosa.const_shape {values = dense<[ 2, 1, 1, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %11 = tosa.tile %10, %t_11 : (tensor<5x10x6x8xi1>, !tosa.shape<4>) -> tensor<10x10x6x16xi1>
    return %1, %7, %11 : tensor<i1>, tensor<12x54x5x44xf32>, tensor<10x10x6x16xi1>
  }
}
