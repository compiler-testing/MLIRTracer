module {
  func.func @main(%arg0: tensor<43x72x80x65xi32>, %arg1: tensor<62x85x100x67xf32>, %arg2: tensor<62x85x100x1xf32>) -> (tensor<62x85x100x67xi1>, tensor<43x72x80x65xi1>, tensor<43x72x80x65xi1>, tensor<43x72x1x65xi1>, tensor<1x85x300x67xf32>, tensor<43x72x80x65xi1>) {
    %0 = tosa.identity %arg0 : (tensor<43x72x80x65xi32>) -> tensor<43x72x80x65xi32>
    %1 = tosa.greater_equal %0, %0 : (tensor<43x72x80x65xi32>, tensor<43x72x80x65xi32>) -> tensor<43x72x80x65xi1>
    %2 = tosa.greater %0, %0 : (tensor<43x72x80x65xi32>, tensor<43x72x80x65xi32>) -> tensor<43x72x80x65xi1>
    %3 = tosa.sub %1, %1 : (tensor<43x72x80x65xi1>, tensor<43x72x80x65xi1>) -> tensor<43x72x80x65xi1>
    %4 = tosa.bitwise_xor %3, %2 : (tensor<43x72x80x65xi1>, tensor<43x72x80x65xi1>) -> tensor<43x72x80x65xi1>
    %5 = tosa.pow %arg1, %arg2 : (tensor<62x85x100x67xf32>, tensor<62x85x100x1xf32>) -> tensor<62x85x100x67xf32>
    %6 = tosa.greater_equal %5, %5 : (tensor<62x85x100x67xf32>, tensor<62x85x100x67xf32>) -> tensor<62x85x100x67xi1>
    %7 = tosa.tanh %5 : (tensor<62x85x100x67xf32>) -> tensor<62x85x100x67xf32>
    %t_8 = tosa.const_shape {values = dense<[ 3, 1, 3, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %8 = tosa.tile %7, %t_8 : (tensor<62x85x100x67xf32>, !tosa.shape<4>) -> tensor<186x85x300x67xf32>
    %9 = tosa.logical_and %1, %1 : (tensor<43x72x80x65xi1>, tensor<43x72x80x65xi1>) -> tensor<43x72x80x65xi1>
    %10 = tosa.logical_xor %4, %1 : (tensor<43x72x80x65xi1>, tensor<43x72x80x65xi1>) -> tensor<43x72x80x65xi1>
    %11 = tosa.reverse %8 {axis = 0 : i32} : (tensor<186x85x300x67xf32>) -> tensor<186x85x300x67xf32>
    %12 = tosa.logical_right_shift %4, %1 : (tensor<43x72x80x65xi1>, tensor<43x72x80x65xi1>) -> tensor<43x72x80x65xi1>
    %13 = tosa.reduce_all %4 {axis = 2 : i32} : (tensor<43x72x80x65xi1>) -> tensor<43x72x1x65xi1>
    %14 = tosa.reduce_min %11 {axis = 0 : i32} : (tensor<186x85x300x67xf32>) -> tensor<1x85x300x67xf32>
    %15 = tosa.bitwise_not %9 : (tensor<43x72x80x65xi1>) -> tensor<43x72x80x65xi1>
    return %6, %10, %12, %13, %14, %15 : tensor<62x85x100x67xi1>, tensor<43x72x80x65xi1>, tensor<43x72x80x65xi1>, tensor<43x72x1x65xi1>, tensor<1x85x300x67xf32>, tensor<43x72x80x65xi1>
  }
}
