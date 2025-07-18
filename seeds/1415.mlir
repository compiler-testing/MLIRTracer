module {
  func.func @main(%arg0: tensor<49x74x16x7xi32>, %arg1: tensor<1x74x1x1xi32>, %arg2: tensor<34x74x26x74x59x56xf32>) -> (tensor<1x74x1x1xi1>, tensor<12x8x1x10x10x11xf32>, tensor<34x74x26x74x59x56xf32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<49x74x16x7xi32>, tensor<1x74x1x1xi32>) -> tensor<49x74x16x7xi32>
    %1 = tosa.reduce_sum %0 {axis = 2 : i32} : (tensor<49x74x16x7xi32>) -> tensor<49x74x1x7xi32>
    %2 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<49x74x1x7xi32>, tensor<49x74x1x7xi32>) -> tensor<49x74x1x7xi32>
    %3 = tosa.equal %2, %2 : (tensor<49x74x1x7xi32>, tensor<49x74x1x7xi32>) -> tensor<49x74x1x7xi1>
    %4 = tosa.ceil %arg2 : (tensor<34x74x26x74x59x56xf32>) -> tensor<34x74x26x74x59x56xf32>
    %5 = tosa.reduce_sum %3 {axis = 0 : i32} : (tensor<49x74x1x7xi1>) -> tensor<1x74x1x7xi1>
    %6 = tosa.log %4 : (tensor<34x74x26x74x59x56xf32>) -> tensor<34x74x26x74x59x56xf32>
    %7 = tosa.reduce_any %5 {axis = 3 : i32} : (tensor<1x74x1x7xi1>) -> tensor<1x74x1x1xi1>
    %s_8_start = tosa.const_shape {values = dense<[ 20, 6, 15, 18, 9, 34 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_8_size = tosa.const_shape {values = dense<[ 12, 8, 1, 10, 10, 11 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %8 = tosa.slice %6, %s_8_start, %s_8_size : (tensor<34x74x26x74x59x56xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<12x8x1x10x10x11xf32>
    %9 = tosa.pow %4, %4 : (tensor<34x74x26x74x59x56xf32>, tensor<34x74x26x74x59x56xf32>) -> tensor<34x74x26x74x59x56xf32>
    return %7, %8, %9 : tensor<1x74x1x1xi1>, tensor<12x8x1x10x10x11xf32>, tensor<34x74x26x74x59x56xf32>
  }
}
