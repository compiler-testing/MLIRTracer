module {
  func.func @main(%arg0: tensor<35x41xi8>, %arg1: tensor<35x1xi8>, %arg2: tensor<42x48xf32>, %arg3: tensor<42x1xf32>) -> (tensor<42x48xf32>, tensor<35x41xi1>, tensor<35x1xi1>, tensor<35x41xi1>, tensor<7x12x6x5xf32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<35x41xi8>, tensor<35x1xi8>) -> tensor<35x41xi1>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<35x41xi1>, tensor<35x41xi1>) -> tensor<35x41xi1>
    %2 = tosa.pow %arg2, %arg3 : (tensor<42x48xf32>, tensor<42x1xf32>) -> tensor<42x48xf32>
    %3 = tosa.logical_or %1, %0 : (tensor<35x41xi1>, tensor<35x41xi1>) -> tensor<35x41xi1>
    %4 = tosa.ceil %2 : (tensor<42x48xf32>) -> tensor<42x48xf32>
    %5 = tosa.bitwise_not %3 : (tensor<35x41xi1>) -> tensor<35x41xi1>
    %r_6 = tosa.const_shape {values = dense<[ 1, 1008, 1, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %6 = tosa.reshape %2, %r_6 : (tensor<42x48xf32>, !tosa.shape<4>) -> tensor<1x1008x1x2xf32>
    %7 = tosa.exp %6 : (tensor<1x1008x1x2xf32>) -> tensor<1x1008x1x2xf32>
    %8 = tosa.reduce_any %3 {axis = 1 : i32} : (tensor<35x41xi1>) -> tensor<35x1xi1>
    %9 = tosa.arithmetic_right_shift %5, %5 {round = true} : (tensor<35x41xi1>, tensor<35x41xi1>) -> tensor<35x41xi1>
    %10 = tosa.bitwise_or %8, %8 : (tensor<35x1xi1>, tensor<35x1xi1>) -> tensor<35x1xi1>
    %11 = tosa.logical_left_shift %3, %5 : (tensor<35x41xi1>, tensor<35x41xi1>) -> tensor<35x41xi1>
    %s_12_start = tosa.const_shape {values = dense<[ 0, 1, 0, 0 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_12_size = tosa.const_shape {values = dense<[ 7, 12, 6, 5 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %12 = tosa.slice %7, %s_12_start, %s_12_size : (tensor<1x1008x1x2xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<7x12x6x5xf32>
    return %4, %9, %10, %11, %12 : tensor<42x48xf32>, tensor<35x41xi1>, tensor<35x1xi1>, tensor<35x41xi1>, tensor<7x12x6x5xf32>
  }
}
