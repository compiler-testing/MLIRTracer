module {
  func.func @main(%arg0: tensor<3x31xf32>) -> (tensor<3x31xf32>, tensor<3x31xf32>, tensor<18x62xi1>, tensor<3x31xf32>, tensor<18x1xi1>, tensor<1x31xf32>, tensor<12x8xf32>, tensor<i32>, tensor<3x31xf32>, tensor<3x31xf32>) {
    %0 = tosa.ceil %arg0 : (tensor<3x31xf32>) -> tensor<3x31xf32>
    %1 = tosa.minimum %0, %0 : (tensor<3x31xf32>, tensor<3x31xf32>) -> tensor<3x31xf32>
    %2 = tosa.minimum %1, %1 : (tensor<3x31xf32>, tensor<3x31xf32>) -> tensor<3x31xf32>
    %3 = tosa.log %2 : (tensor<3x31xf32>) -> tensor<3x31xf32>
    %4 = tosa.ceil %3 : (tensor<3x31xf32>) -> tensor<3x31xf32>
    %5 = tosa.exp %4 : (tensor<3x31xf32>) -> tensor<3x31xf32>
    %6 = tosa.add %5, %0 : (tensor<3x31xf32>, tensor<3x31xf32>) -> tensor<3x31xf32>
    %7 = tosa.concat %6, %2 {axis = 0 : i32} : (tensor<3x31xf32>, tensor<3x31xf32>) -> tensor<6x31xf32>
    %8 = tosa.greater %7, %7 : (tensor<6x31xf32>, tensor<6x31xf32>) -> tensor<6x31xi1>
    %9 = tosa.pow %3, %3 : (tensor<3x31xf32>, tensor<3x31xf32>) -> tensor<3x31xf32>
    %10 = tosa.tanh %4 : (tensor<3x31xf32>) -> tensor<3x31xf32>
    %t_11 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %11 = tosa.tile %8, %t_11 : (tensor<6x31xi1>, !tosa.shape<2>) -> tensor<18x62xi1>
    %12 = tosa.reduce_sum %11 {axis = 0 : i32} : (tensor<18x62xi1>) -> tensor<1x62xi1>
    %13 = tosa.floor %0 : (tensor<3x31xf32>) -> tensor<3x31xf32>
    %14 = tosa.argmax %12 {axis = 0 : i32} : (tensor<1x62xi1>) -> tensor<62xi32>
    %15 = tosa.arithmetic_right_shift %14, %14 {round = true} : (tensor<62xi32>, tensor<62xi32>) -> tensor<62xi32>
    %16 = tosa.tanh %0 : (tensor<3x31xf32>) -> tensor<3x31xf32>
    %17 = tosa.logical_not %11 : (tensor<18x62xi1>) -> tensor<18x62xi1>
    %18 = tosa.argmax %15 {axis = 0 : i32} : (tensor<62xi32>) -> tensor<i32>
    %19 = tosa.add %18, %18 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %20 = tosa.exp %2 : (tensor<3x31xf32>) -> tensor<3x31xf32>
    %21 = tosa.reduce_all %11 {axis = 1 : i32} : (tensor<18x62xi1>) -> tensor<18x1xi1>
    %22 = tosa.add %19, %19 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %23 = tosa.reduce_max %16 {axis = 0 : i32} : (tensor<3x31xf32>) -> tensor<1x31xf32>
    %s_24_start = tosa.const_shape {values = dense<[ 0, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_24_size = tosa.const_shape {values = dense<[ 12, 8 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %24 = tosa.slice %0, %s_24_start, %s_24_size : (tensor<3x31xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<12x8xf32>
    %25 = tosa.clamp %22 {min_val = 23 : i32, max_val = 40 : i32} : (tensor<i32>) -> tensor<i32>
    %26 = tosa.exp %0 : (tensor<3x31xf32>) -> tensor<3x31xf32>
    %27 = tosa.tanh %13 : (tensor<3x31xf32>) -> tensor<3x31xf32>
    return %9, %10, %17, %20, %21, %23, %24, %25, %26, %27 : tensor<3x31xf32>, tensor<3x31xf32>, tensor<18x62xi1>, tensor<3x31xf32>, tensor<18x1xi1>, tensor<1x31xf32>, tensor<12x8xf32>, tensor<i32>, tensor<3x31xf32>, tensor<3x31xf32>
  }
}
