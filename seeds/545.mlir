module {
  func.func @main(%arg0: tensor<7x71xi32>, %arg1: tensor<1x1xi32>, %arg2: tensor<18x69x95xf32>, %arg3: tensor<86x83x96xi1>, %arg4: tensor<1x83x1xi1>) -> (tensor<7x71xi32>, tensor<86x83x96xi1>, tensor<2x2x1xf32>, tensor<18x69x190xf32>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<7x71xi32>, tensor<1x1xi32>) -> tensor<7x71xi32>
    %1 = tosa.bitwise_or %0, %0 : (tensor<7x71xi32>, tensor<7x71xi32>) -> tensor<7x71xi32>
    %2 = tosa.bitwise_xor %1, %0 : (tensor<7x71xi32>, tensor<7x71xi32>) -> tensor<7x71xi32>
    %3 = tosa.logical_right_shift %2, %1 : (tensor<7x71xi32>, tensor<7x71xi32>) -> tensor<7x71xi32>
    %4 = tosa.rsqrt %arg2 : (tensor<18x69x95xf32>) -> tensor<18x69x95xf32>
    %5 = tosa.identity %4 : (tensor<18x69x95xf32>) -> tensor<18x69x95xf32>
    %6 = tosa.clamp %5 {min_val = -5.800000e+01 : f32, max_val = 5.600000e+01 : f32} : (tensor<18x69x95xf32>) -> tensor<18x69x95xf32>
    %7 = "tosa.const"() {values = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
    %8 = tosa.transpose %6 {perms = array<i32: 0, 1, 2>} : (tensor<18x69x95xf32>) -> tensor<18x69x95xf32>
    %9 = tosa.sub %8, %8 : (tensor<18x69x95xf32>, tensor<18x69x95xf32>) -> tensor<18x69x95xf32>
    %10 = tosa.concat %9, %6 {axis = 2 : i32} : (tensor<18x69x95xf32>, tensor<18x69x95xf32>) -> tensor<18x69x190xf32>
    %11 = tosa.logical_or %arg3, %arg4 : (tensor<86x83x96xi1>, tensor<1x83x1xi1>) -> tensor<86x83x96xi1>
    %12 = tosa.logical_and %11, %11 : (tensor<86x83x96xi1>, tensor<86x83x96xi1>) -> tensor<86x83x96xi1>
    %s_13_start = tosa.const_shape {values = dense<[ 4, 18, 10 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_13_size = tosa.const_shape {values = dense<[ 2, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %13 = tosa.slice %10, %s_13_start, %s_13_size : (tensor<18x69x190xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x2x1xf32>
    %14 = tosa.ceil %13 : (tensor<2x2x1xf32>) -> tensor<2x2x1xf32>
    %15 = tosa.reverse %14 {axis = 1 : i32} : (tensor<2x2x1xf32>) -> tensor<2x2x1xf32>
    %16 = tosa.ceil %10 : (tensor<18x69x190xf32>) -> tensor<18x69x190xf32>
    return %3, %12, %15, %16 : tensor<7x71xi32>, tensor<86x83x96xi1>, tensor<2x2x1xf32>, tensor<18x69x190xf32>
  }
}
