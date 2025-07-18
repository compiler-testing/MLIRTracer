module {
  func.func @main(%arg0: tensor<8x79x64x47x33x82xf32>, %arg1: tensor<8x23x23x25xi1>, %arg2: tensor<1x23x23x25xi1>) -> (tensor<3x9x8x7x7x5xi1>, tensor<1x1x1x1xi1>, tensor<3x9x8x7x7x5xf32>, tensor<23x25xi32>, tensor<8x23x25xi1>) {
    %s_0_start = tosa.const_shape {values = dense<[ 4, 8, 4, 4, 7, 8 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_0_size = tosa.const_shape {values = dense<[ 3, 9, 8, 7, 7, 5 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<8x79x64x47x33x82xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<3x9x8x7x7x5xf32>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<8x23x23x25xi1>, tensor<1x23x23x25xi1>) -> tensor<8x23x23x25xi1>
    %2 = tosa.logical_not %1 : (tensor<8x23x23x25xi1>) -> tensor<8x23x23x25xi1>
    %3 = tosa.logical_or %1, %1 : (tensor<8x23x23x25xi1>, tensor<8x23x23x25xi1>) -> tensor<8x23x23x25xi1>
    %4 = tosa.greater %0, %0 : (tensor<3x9x8x7x7x5xf32>, tensor<3x9x8x7x7x5xf32>) -> tensor<3x9x8x7x7x5xi1>
    %5 = tosa.reduce_sum %2 {axis = 1 : i32} : (tensor<8x23x23x25xi1>) -> tensor<8x1x23x25xi1>
    %6 = tosa.floor %0 : (tensor<3x9x8x7x7x5xf32>) -> tensor<3x9x8x7x7x5xf32>
    %7 = tosa.argmax %3 {axis = 1 : i32} : (tensor<8x23x23x25xi1>) -> tensor<8x23x25xi32>
    %8 = tosa.reduce_any %5 {axis = 3 : i32} : (tensor<8x1x23x25xi1>) -> tensor<8x1x23x1xi1>
    %9 = tosa.bitwise_and %7, %7 : (tensor<8x23x25xi32>, tensor<8x23x25xi32>) -> tensor<8x23x25xi32>
    %10 = tosa.reduce_all %8 {axis = 2 : i32} : (tensor<8x1x23x1xi1>) -> tensor<8x1x1x1xi1>
    %11 = tosa.logical_or %10, %10 : (tensor<8x1x1x1xi1>, tensor<8x1x1x1xi1>) -> tensor<8x1x1x1xi1>
    %12 = tosa.reduce_all %11 {axis = 0 : i32} : (tensor<8x1x1x1xi1>) -> tensor<1x1x1x1xi1>
    %13 = tosa.equal %9, %7 : (tensor<8x23x25xi32>, tensor<8x23x25xi32>) -> tensor<8x23x25xi1>
    %14 = tosa.reciprocal %6 : (tensor<3x9x8x7x7x5xf32>) -> tensor<3x9x8x7x7x5xf32>
    %15 = tosa.logical_xor %13, %13 : (tensor<8x23x25xi1>, tensor<8x23x25xi1>) -> tensor<8x23x25xi1>
    %16 = tosa.argmax %9 {axis = 0 : i32} : (tensor<8x23x25xi32>) -> tensor<23x25xi32>
    %17 = tosa.arithmetic_right_shift %15, %15 {round = false} : (tensor<8x23x25xi1>, tensor<8x23x25xi1>) -> tensor<8x23x25xi1>
    return %4, %12, %14, %16, %17 : tensor<3x9x8x7x7x5xi1>, tensor<1x1x1x1xi1>, tensor<3x9x8x7x7x5xf32>, tensor<23x25xi32>, tensor<8x23x25xi1>
  }
}
