module {
  func.func @main(%arg0: tensor<83x21x11x9xi32>, %arg1: tensor<1x1x11x9xi32>, %arg2: tensor<36x58x94xi1>, %arg3: tensor<25x91x11xf32>) -> (tensor<83x21x11x9xi1>, tensor<25x91x11xf32>, tensor<83x42x11x9xi1>, tensor<36x1x94xi1>, tensor<36x3x188xi1>, tensor<36x1x94xi1>, tensor<36x1x94xi1>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<83x21x11x9xi32>, tensor<1x1x11x9xi32>) -> tensor<83x21x11x9xi32>
    %1 = tosa.reduce_all %arg2 {axis = 1 : i32} : (tensor<36x58x94xi1>) -> tensor<36x1x94xi1>
    %2 = tosa.bitwise_not %1 : (tensor<36x1x94xi1>) -> tensor<36x1x94xi1>
    %3 = tosa.greater_equal %0, %0 : (tensor<83x21x11x9xi32>, tensor<83x21x11x9xi32>) -> tensor<83x21x11x9xi1>
    %4 = tosa.equal %0, %0 : (tensor<83x21x11x9xi32>, tensor<83x21x11x9xi32>) -> tensor<83x21x11x9xi1>
    %5 = tosa.rsqrt %arg3 : (tensor<25x91x11xf32>) -> tensor<25x91x11xf32>
    %6 = tosa.concat %3, %3 {axis = 1 : i32} : (tensor<83x21x11x9xi1>, tensor<83x21x11x9xi1>) -> tensor<83x42x11x9xi1>
    %7 = tosa.bitwise_and %6, %6 : (tensor<83x42x11x9xi1>, tensor<83x42x11x9xi1>) -> tensor<83x42x11x9xi1>
    %8 = tosa.arithmetic_right_shift %7, %7 {round = true} : (tensor<83x42x11x9xi1>, tensor<83x42x11x9xi1>) -> tensor<83x42x11x9xi1>
    %9 = tosa.logical_not %2 : (tensor<36x1x94xi1>) -> tensor<36x1x94xi1>
    %10 = tosa.sub %9, %2 : (tensor<36x1x94xi1>, tensor<36x1x94xi1>) -> tensor<36x1x94xi1>
    %t_11 = tosa.const_shape {values = dense<[ 1, 3, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %11 = tosa.tile %9, %t_11 : (tensor<36x1x94xi1>, !tosa.shape<3>) -> tensor<36x3x188xi1>
    %12 = tosa.abs %9 : (tensor<36x1x94xi1>) -> tensor<36x1x94xi1>
    %13 = tosa.reduce_sum %9 {axis = 1 : i32} : (tensor<36x1x94xi1>) -> tensor<36x1x94xi1>
    return %4, %5, %8, %10, %11, %12, %13 : tensor<83x21x11x9xi1>, tensor<25x91x11xf32>, tensor<83x42x11x9xi1>, tensor<36x1x94xi1>, tensor<36x3x188xi1>, tensor<36x1x94xi1>, tensor<36x1x94xi1>
  }
}
