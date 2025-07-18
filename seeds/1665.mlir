module {
  func.func @main(%arg0: tensor<24x60x25xi64>, %arg1: tensor<1x1x1xi64>, %arg2: tensor<87x11x100x5x4xf32>, %arg3: tensor<100x66x35xi1>, %arg4: tensor<1x66x35xi1>, %arg5: tensor<37x76x3x35xi32>, %arg6: tensor<37x76x3x1xi32>) -> (tensor<87x11x100x5x4xf32>, tensor<72x60x50xi64>, tensor<24x60x25xi64>, tensor<1xi1>, tensor<37x76x3x35xi32>, tensor<4x5x11x100x174xi1>, tensor<1x66x1xi1>, tensor<4x5x11x100x87xf32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<24x60x25xi64>, tensor<1x1x1xi64>) -> tensor<24x60x25xi64>
    %1 = tosa.tanh %arg2 : (tensor<87x11x100x5x4xf32>) -> tensor<87x11x100x5x4xf32>
    %2 = tosa.ceil %1 : (tensor<87x11x100x5x4xf32>) -> tensor<87x11x100x5x4xf32>
    %t_3 = tosa.const_shape {values = dense<[ 3, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.tile %0, %t_3 : (tensor<24x60x25xi64>, !tosa.shape<3>) -> tensor<72x60x50xi64>
    %4 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %5 = tosa.transpose %1 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<87x11x100x5x4xf32>) -> tensor<4x5x11x100x87xf32>
    %6 = tosa.logical_right_shift %0, %0 : (tensor<24x60x25xi64>, tensor<24x60x25xi64>) -> tensor<24x60x25xi64>
    %7 = tosa.logical_and %arg3, %arg4 : (tensor<100x66x35xi1>, tensor<1x66x35xi1>) -> tensor<100x66x35xi1>
    %8 = tosa.logical_xor %7, %7 : (tensor<100x66x35xi1>, tensor<100x66x35xi1>) -> tensor<100x66x35xi1>
    %9 = tosa.reduce_any %8 {axis = 2 : i32} : (tensor<100x66x35xi1>) -> tensor<100x66x1xi1>
    %10 = tosa.concat %5, %5 {axis = 4 : i32} : (tensor<4x5x11x100x87xf32>, tensor<4x5x11x100x87xf32>) -> tensor<4x5x11x100x174xf32>
    %11 = tosa.logical_not %9 : (tensor<100x66x1xi1>) -> tensor<100x66x1xi1>
    %r_12 = tosa.const_shape {values = dense<[ 6600 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %12 = tosa.reshape %11, %r_12 : (tensor<100x66x1xi1>, !tosa.shape<1>) -> tensor<6600xi1>
    %13 = tosa.clz %12 : (tensor<6600xi1>) -> tensor<6600xi1>
    %14 = tosa.reduce_sum %13 {axis = 0 : i32} : (tensor<6600xi1>) -> tensor<1xi1>
    %15 = tosa.ceil %5 : (tensor<4x5x11x100x87xf32>) -> tensor<4x5x11x100x87xf32>
    %16 = tosa.intdiv %arg5, %arg6 : (tensor<37x76x3x35xi32>, tensor<37x76x3x1xi32>) -> tensor<37x76x3x35xi32>
    %17 = tosa.greater_equal %10, %10 : (tensor<4x5x11x100x174xf32>, tensor<4x5x11x100x174xf32>) -> tensor<4x5x11x100x174xi1>
    %18 = tosa.reduce_all %11 {axis = 0 : i32} : (tensor<100x66x1xi1>) -> tensor<1x66x1xi1>
    %19 = tosa.sub %15, %5 : (tensor<4x5x11x100x87xf32>, tensor<4x5x11x100x87xf32>) -> tensor<4x5x11x100x87xf32>
    return %2, %3, %6, %14, %16, %17, %18, %19 : tensor<87x11x100x5x4xf32>, tensor<72x60x50xi64>, tensor<24x60x25xi64>, tensor<1xi1>, tensor<37x76x3x35xi32>, tensor<4x5x11x100x174xi1>, tensor<1x66x1xi1>, tensor<4x5x11x100x87xf32>
  }
}
