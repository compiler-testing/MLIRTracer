module {
  func.func @main(%arg0: tensor<4x58x70xi32>, %arg1: tensor<1x1x70xi32>, %arg2: tensor<58x17x3x18x6xf32>) -> (tensor<5x3248xi1>, tensor<58x17x3x18x6xf32>, tensor<3248x1xi1>, tensor<5x1xi32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<4x58x70xi32>, tensor<1x1x70xi32>) -> tensor<4x58x70xi32>
    %r_1 = tosa.const_shape {values = dense<[ 2, 5, 1624, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.reshape %0, %r_1 : (tensor<4x58x70xi32>, !tosa.shape<4>) -> tensor<2x5x1624x1xi32>
    %r_2 = tosa.const_shape {values = dense<[ 5, 3248 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %1, %r_2 : (tensor<2x5x1624x1xi32>, !tosa.shape<2>) -> tensor<5x3248xi32>
    %3 = tosa.greater %2, %2 : (tensor<5x3248xi32>, tensor<5x3248xi32>) -> tensor<5x3248xi1>
    %4 = tosa.logical_not %3 : (tensor<5x3248xi1>) -> tensor<5x3248xi1>
    %5 = tosa.reduce_all %4 {axis = 0 : i32} : (tensor<5x3248xi1>) -> tensor<1x3248xi1>
    %6 = "tosa.const"() {values = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %7 = tosa.transpose %5 {perms = array<i32: 1, 0>} : (tensor<1x3248xi1>) -> tensor<3248x1xi1>
    %8 = tosa.sigmoid %arg2 : (tensor<58x17x3x18x6xf32>) -> tensor<58x17x3x18x6xf32>
    %9 = tosa.maximum %8, %8 : (tensor<58x17x3x18x6xf32>, tensor<58x17x3x18x6xf32>) -> tensor<58x17x3x18x6xf32>
    %10 = tosa.bitwise_or %4, %4 : (tensor<5x3248xi1>, tensor<5x3248xi1>) -> tensor<5x3248xi1>
    %11 = tosa.exp %9 : (tensor<58x17x3x18x6xf32>) -> tensor<58x17x3x18x6xf32>
    %12 = tosa.reduce_any %7 {axis = 1 : i32} : (tensor<3248x1xi1>) -> tensor<3248x1xi1>
    %13 = tosa.reduce_sum %2 {axis = 1 : i32} : (tensor<5x3248xi32>) -> tensor<5x1xi32>
    return %10, %11, %12, %13 : tensor<5x3248xi1>, tensor<58x17x3x18x6xf32>, tensor<3248x1xi1>, tensor<5x1xi32>
  }
}
