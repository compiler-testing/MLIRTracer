module {
  func.func @main(%arg0: tensor<21xi1>, %arg1: tensor<21xi1>, %arg2: tensor<59x41x22x41xf32>) -> (tensor<11x9x2x9xf32>, tensor<1x4x9xi1>, tensor<1xi1>, tensor<1x1x1xi1>, tensor<1xi1>, tensor<1298x41x41xf32>, tensor<9x4x9xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<21xi1>, tensor<21xi1>) -> tensor<21xi1>
    %t_1 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.tile %0, %t_1 : (tensor<21xi1>, !tosa.shape<1>) -> tensor<63xi1>
    %2 = tosa.reduce_any %1 {axis = 0 : i32} : (tensor<63xi1>) -> tensor<1xi1>
    %3 = tosa.sub %2, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.reduce_any %3 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_5_size = tosa.const_shape {values = dense<[ 11 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<1xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<11xi1>
    %6 = tosa.logical_or %5, %5 : (tensor<11xi1>, tensor<11xi1>) -> tensor<11xi1>
    %7 = tosa.reduce_max %6 {axis = 0 : i32} : (tensor<11xi1>) -> tensor<1xi1>
    %8 = tosa.bitwise_or %7, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %r_9 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = tosa.reshape %8, %r_9 : (tensor<1xi1>, !tosa.shape<3>) -> tensor<1x1x1xi1>
    %10 = tosa.clz %9 : (tensor<1x1x1xi1>) -> tensor<1x1x1xi1>
    %11 = tosa.log %arg2 : (tensor<59x41x22x41xf32>) -> tensor<59x41x22x41xf32>
    %s_12_start = tosa.const_shape {values = dense<[ 0, 0, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_12_size = tosa.const_shape {values = dense<[ 9, 4, 9 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %12 = tosa.slice %10, %s_12_start, %s_12_size : (tensor<1x1x1xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<9x4x9xi1>
    %13 = tosa.reduce_all %12 {axis = 0 : i32} : (tensor<9x4x9xi1>) -> tensor<1x4x9xi1>
    %14 = tosa.floor %11 : (tensor<59x41x22x41xf32>) -> tensor<59x41x22x41xf32>
    %15 = tosa.reduce_all %9 {axis = 2 : i32} : (tensor<1x1x1xi1>) -> tensor<1x1x1xi1>
    %s_16_start = tosa.const_shape {values = dense<[ 6, 32, 20, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_16_size = tosa.const_shape {values = dense<[ 11, 9, 2, 9 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %16 = tosa.slice %11, %s_16_start, %s_16_size : (tensor<59x41x22x41xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<11x9x2x9xf32>
    %17 = tosa.bitwise_and %13, %13 : (tensor<1x4x9xi1>, tensor<1x4x9xi1>) -> tensor<1x4x9xi1>
    %18 = tosa.logical_not %2 : (tensor<1xi1>) -> tensor<1xi1>
    %19 = tosa.maximum %14, %11 : (tensor<59x41x22x41xf32>, tensor<59x41x22x41xf32>) -> tensor<59x41x22x41xf32>
    %20 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %21 = tosa.transpose %19 {perms = array<i32: 0, 1, 3, 2>} : (tensor<59x41x22x41xf32>) -> tensor<59x41x41x22xf32>
    %22 = tosa.bitwise_or %15, %15 : (tensor<1x1x1xi1>, tensor<1x1x1xi1>) -> tensor<1x1x1xi1>
    %23 = tosa.floor %21 : (tensor<59x41x41x22xf32>) -> tensor<59x41x41x22xf32>
    %24 = tosa.bitwise_or %7, %7 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %r_25 = tosa.const_shape {values = dense<[ 1298, 41, 41 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %25 = tosa.reshape %23, %r_25 : (tensor<59x41x41x22xf32>, !tosa.shape<3>) -> tensor<1298x41x41xf32>
    %26 = tosa.bitwise_or %12, %12 : (tensor<9x4x9xi1>, tensor<9x4x9xi1>) -> tensor<9x4x9xi1>
    return %16, %17, %18, %22, %24, %25, %26 : tensor<11x9x2x9xf32>, tensor<1x4x9xi1>, tensor<1xi1>, tensor<1x1x1xi1>, tensor<1xi1>, tensor<1298x41x41xf32>, tensor<9x4x9xi1>
  }
}
