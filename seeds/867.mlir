module {
  func.func @main(%arg0: tensor<86x31x83xi16>, %arg1: tensor<86x83x27xi16>, %arg2: tensor<29x29x79x21xf32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> (tensor<86x31x27xi16>, tensor<i1>, tensor<i32>, tensor<29x1x79x21xf32>, tensor<29x29x79x21xf32>, tensor<29x1x79x21xf32>, tensor<29x1x79x21xf32>, tensor<29x1x79x21xf32>, tensor<1x1xi32>, tensor<1x1x11x4xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<86x31x83xi16>, tensor<86x83x27xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<86x31x27xi16>
    %1 = tosa.rsqrt %arg2 : (tensor<29x29x79x21xf32>) -> tensor<29x29x79x21xf32>
    %2 = tosa.greater_equal %1, %1 : (tensor<29x29x79x21xf32>, tensor<29x29x79x21xf32>) -> tensor<29x29x79x21xi1>
    %3 = tosa.intdiv %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<29x29x79x21xf32>) -> tensor<29x1x79x21xf32>
    %5 = tosa.bitwise_not %3 : (tensor<i32>) -> tensor<i32>
    %6 = tosa.reciprocal %4 : (tensor<29x1x79x21xf32>) -> tensor<29x1x79x21xf32>
    %7 = tosa.minimum %4, %6 : (tensor<29x1x79x21xf32>, tensor<29x1x79x21xf32>) -> tensor<29x1x79x21xf32>
    %8 = tosa.greater %5, %5 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = tosa.exp %6 : (tensor<29x1x79x21xf32>) -> tensor<29x1x79x21xf32>
    %10 = tosa.reduce_product %2 {axis = 1 : i32} : (tensor<29x29x79x21xi1>) -> tensor<29x1x79x21xi1>
    %11 = tosa.exp %6 : (tensor<29x1x79x21xf32>) -> tensor<29x1x79x21xf32>
    %12 = tosa.bitwise_xor %3, %3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %13 = tosa.logical_left_shift %10, %10 : (tensor<29x1x79x21xi1>, tensor<29x1x79x21xi1>) -> tensor<29x1x79x21xi1>
    %14 = tosa.bitwise_not %13 : (tensor<29x1x79x21xi1>) -> tensor<29x1x79x21xi1>
    %s_15_start = tosa.const_shape {values = dense<[ 12, 0, 10, 13 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_15_size = tosa.const_shape {values = dense<[ 7, 11, 4, 8 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %15 = tosa.slice %14, %s_15_start, %s_15_size : (tensor<29x1x79x21xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<7x11x4x8xi1>
    %16 = tosa.minimum %4, %6 : (tensor<29x1x79x21xf32>, tensor<29x1x79x21xf32>) -> tensor<29x1x79x21xf32>
    %17 = tosa.minimum %4, %9 : (tensor<29x1x79x21xf32>, tensor<29x1x79x21xf32>) -> tensor<29x1x79x21xf32>
    %18 = tosa.logical_right_shift %15, %15 : (tensor<7x11x4x8xi1>, tensor<7x11x4x8xi1>) -> tensor<7x11x4x8xi1>
    %19 = tosa.reduce_max %18 {axis = 3 : i32} : (tensor<7x11x4x8xi1>) -> tensor<7x11x4x1xi1>
    %20 = tosa.floor %1 : (tensor<29x29x79x21xf32>) -> tensor<29x29x79x21xf32>
    %21 = tosa.logical_xor %19, %19 : (tensor<7x11x4x1xi1>, tensor<7x11x4x1xi1>) -> tensor<7x11x4x1xi1>
    %22 = tosa.minimum %7, %7 : (tensor<29x1x79x21xf32>, tensor<29x1x79x21xf32>) -> tensor<29x1x79x21xf32>
    %23 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %24 = tosa.transpose %21 {perms = array<i32: 0, 3, 1, 2>} : (tensor<7x11x4x1xi1>) -> tensor<7x1x11x4xi1>
    %25 = tosa.rsqrt %11 : (tensor<29x1x79x21xf32>) -> tensor<29x1x79x21xf32>
    %26 = tosa.reduce_product %24 {axis = 0 : i32} : (tensor<7x1x11x4xi1>) -> tensor<1x1x11x4xi1>
    %27 = tosa.pow %11, %17 : (tensor<29x1x79x21xf32>, tensor<29x1x79x21xf32>) -> tensor<29x1x79x21xf32>
    %r_28 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %28 = tosa.reshape %3, %r_28 : (tensor<i32>, !tosa.shape<2>) -> tensor<1x1xi32>
    %29 = tosa.reduce_max %26 {axis = 1 : i32} : (tensor<1x1x11x4xi1>) -> tensor<1x1x11x4xi1>
    return %0, %8, %12, %16, %20, %22, %25, %27, %28, %29 : tensor<86x31x27xi16>, tensor<i1>, tensor<i32>, tensor<29x1x79x21xf32>, tensor<29x29x79x21xf32>, tensor<29x1x79x21xf32>, tensor<29x1x79x21xf32>, tensor<29x1x79x21xf32>, tensor<1x1xi32>, tensor<1x1x11x4xi1>
  }
}
