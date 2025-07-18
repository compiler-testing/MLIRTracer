module {
  func.func @main(%arg0: tensor<31xi64>, %arg1: tensor<95x33x80x48xi1>, %arg2: tensor<95x1x80x48xi1>, %arg3: tensor<57x39xf32>) -> (tensor<186xi64>, tensor<95x1x80x48xi1>, tensor<93xi1>, tensor<57x39xi1>, tensor<57x39xi1>, tensor<95x66x80x48xi1>, tensor<95x33x80x48xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<31xi64>, !tosa.shape<1>) -> tensor<93xi64>
    %1 = tosa.sub %0, %0 : (tensor<93xi64>, tensor<93xi64>) -> tensor<93xi64>
    %t_2 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.tile %1, %t_2 : (tensor<93xi64>, !tosa.shape<1>) -> tensor<186xi64>
    %3 = tosa.logical_or %arg1, %arg2 : (tensor<95x33x80x48xi1>, tensor<95x1x80x48xi1>) -> tensor<95x33x80x48xi1>
    %4 = tosa.reduce_any %3 {axis = 1 : i32} : (tensor<95x33x80x48xi1>) -> tensor<95x1x80x48xi1>
    %5 = tosa.greater_equal %0, %0 : (tensor<93xi64>, tensor<93xi64>) -> tensor<93xi1>
    %6 = tosa.rsqrt %arg3 : (tensor<57x39xf32>) -> tensor<57x39xf32>
    %7 = tosa.greater %6, %6 : (tensor<57x39xf32>, tensor<57x39xf32>) -> tensor<57x39xi1>
    %8 = tosa.greater_equal %6, %6 : (tensor<57x39xf32>, tensor<57x39xf32>) -> tensor<57x39xi1>
    %9 = tosa.concat %3, %3 {axis = 1 : i32} : (tensor<95x33x80x48xi1>, tensor<95x33x80x48xi1>) -> tensor<95x66x80x48xi1>
    %10 = tosa.logical_xor %3, %3 : (tensor<95x33x80x48xi1>, tensor<95x33x80x48xi1>) -> tensor<95x33x80x48xi1>
    %11 = tosa.logical_left_shift %10, %3 : (tensor<95x33x80x48xi1>, tensor<95x33x80x48xi1>) -> tensor<95x33x80x48xi1>
    %12 = tosa.arithmetic_right_shift %11, %10 {round = true} : (tensor<95x33x80x48xi1>, tensor<95x33x80x48xi1>) -> tensor<95x33x80x48xi1>
    return %2, %4, %5, %7, %8, %9, %12 : tensor<186xi64>, tensor<95x1x80x48xi1>, tensor<93xi1>, tensor<57x39xi1>, tensor<57x39xi1>, tensor<95x66x80x48xi1>, tensor<95x33x80x48xi1>
  }
}
