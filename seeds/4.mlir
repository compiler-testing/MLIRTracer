module {
  func.func @main(%arg0: tensor<23x79x91x1x38xf32>, %arg1: tensor<100x57x41xf32>) -> (tensor<1x41xi32>, tensor<23x79x91x1x38xi1>, tensor<1x57x41xf32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<23x79x91x1x38xf32>) -> tensor<23x79x91x1x38xf32>
    %1 = tosa.ceil %0 : (tensor<23x79x91x1x38xf32>) -> tensor<23x79x91x1x38xf32>
    %2 = tosa.minimum %1, %1 : (tensor<23x79x91x1x38xf32>, tensor<23x79x91x1x38xf32>) -> tensor<23x79x91x1x38xf32>
    %3 = tosa.rsqrt %2 : (tensor<23x79x91x1x38xf32>) -> tensor<23x79x91x1x38xf32>
    %4 = tosa.reduce_min %arg1 {axis = 0 : i32} : (tensor<100x57x41xf32>) -> tensor<1x57x41xf32>
    %5 = tosa.tanh %3 : (tensor<23x79x91x1x38xf32>) -> tensor<23x79x91x1x38xf32>
    %6 = tosa.rsqrt %5 : (tensor<23x79x91x1x38xf32>) -> tensor<23x79x91x1x38xf32>
    %7 = tosa.greater_equal %6, %3 : (tensor<23x79x91x1x38xf32>, tensor<23x79x91x1x38xf32>) -> tensor<23x79x91x1x38xi1>
    %t_8 = tosa.const_shape {values = dense<[ 2, 3, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %8 = tosa.tile %4, %t_8 : (tensor<1x57x41xf32>, !tosa.shape<3>) -> tensor<2x171x41xf32>
    %9 = tosa.reduce_max %8 {axis = 0 : i32} : (tensor<2x171x41xf32>) -> tensor<1x171x41xf32>
    %10 = tosa.argmax %9 {axis = 1 : i32} : (tensor<1x171x41xf32>) -> tensor<1x41xi32>
    %11 = tosa.logical_left_shift %10, %10 : (tensor<1x41xi32>, tensor<1x41xi32>) -> tensor<1x41xi32>
    %12 = tosa.logical_xor %7, %7 : (tensor<23x79x91x1x38xi1>, tensor<23x79x91x1x38xi1>) -> tensor<23x79x91x1x38xi1>
    %13 = tosa.minimum %4, %4 : (tensor<1x57x41xf32>, tensor<1x57x41xf32>) -> tensor<1x57x41xf32>
    return %11, %12, %13 : tensor<1x41xi32>, tensor<23x79x91x1x38xi1>, tensor<1x57x41xf32>
  }
}
