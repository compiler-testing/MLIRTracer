module {
  func.func @main(%arg0: tensor<39x49xf32>, %arg1: tensor<55xi1>, %arg2: tensor<1xi1>) -> (tensor<7x273xi1>, tensor<39x49xf32>, tensor<7x273xf32>, tensor<5x7xi1>, tensor<i32>, tensor<8x1xi1>, tensor<55xi1>, tensor<7x273xf32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<39x49xf32>) -> tensor<39x49xf32>
    %r_1 = tosa.const_shape {values = dense<[ 7, 273 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %0, %r_1 : (tensor<39x49xf32>, !tosa.shape<2>) -> tensor<7x273xf32>
    %2 = tosa.ceil %1 : (tensor<7x273xf32>) -> tensor<7x273xf32>
    %3 = tosa.logical_xor %arg1, %arg2 : (tensor<55xi1>, tensor<1xi1>) -> tensor<55xi1>
    %4 = tosa.bitwise_not %3 : (tensor<55xi1>) -> tensor<55xi1>
    %5 = tosa.exp %1 : (tensor<7x273xf32>) -> tensor<7x273xf32>
    %6 = tosa.arithmetic_right_shift %4, %3 {round = false} : (tensor<55xi1>, tensor<55xi1>) -> tensor<55xi1>
    %7 = tosa.equal %2, %2 : (tensor<7x273xf32>, tensor<7x273xf32>) -> tensor<7x273xi1>
    %8 = tosa.pow %0, %0 : (tensor<39x49xf32>, tensor<39x49xf32>) -> tensor<39x49xf32>
    %9 = tosa.sub %6, %3 : (tensor<55xi1>, tensor<55xi1>) -> tensor<55xi1>
    %10 = tosa.reduce_max %9 {axis = 0 : i32} : (tensor<55xi1>) -> tensor<1xi1>
    %11 = tosa.clz %10 : (tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.argmax %3 {axis = 0 : i32} : (tensor<55xi1>) -> tensor<i32>
    %13 = tosa.rsqrt %2 : (tensor<7x273xf32>) -> tensor<7x273xf32>
    %14 = tosa.reduce_sum %11 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %r_15 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %15 = tosa.reshape %14, %r_15 : (tensor<1xi1>, !tosa.shape<2>) -> tensor<1x1xi1>
    %16 = tosa.clamp %12 {min_val = -48 : i32, max_val = 73 : i32} : (tensor<i32>) -> tensor<i32>
    %s_17_start = tosa.const_shape {values = dense<[ 0, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_17_size = tosa.const_shape {values = dense<[ 8, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %17 = tosa.slice %15, %s_17_start, %s_17_size : (tensor<1x1xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<8x1xi1>
    %t_18 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %18 = tosa.tile %17, %t_18 : (tensor<8x1xi1>, !tosa.shape<2>) -> tensor<24x2xi1>
    %19 = tosa.bitwise_or %18, %18 : (tensor<24x2xi1>, tensor<24x2xi1>) -> tensor<24x2xi1>
    %s_20_start = tosa.const_shape {values = dense<[ 17, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_20_size = tosa.const_shape {values = dense<[ 5, 7 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %20 = tosa.slice %19, %s_20_start, %s_20_size : (tensor<24x2xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<5x7xi1>
    %21 = tosa.intdiv %16, %16 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %22 = tosa.reduce_product %17 {axis = 1 : i32} : (tensor<8x1xi1>) -> tensor<8x1xi1>
    %23 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %24 = tosa.transpose %3 {perms = array<i32: 0>} : (tensor<55xi1>) -> tensor<55xi1>
    %25 = tosa.ceil %5 : (tensor<7x273xf32>) -> tensor<7x273xf32>
    return %7, %8, %13, %20, %21, %22, %24, %25 : tensor<7x273xi1>, tensor<39x49xf32>, tensor<7x273xf32>, tensor<5x7xi1>, tensor<i32>, tensor<8x1xi1>, tensor<55xi1>, tensor<7x273xf32>
  }
}
