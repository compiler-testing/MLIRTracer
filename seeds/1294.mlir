module {
  func.func @main(%arg0: tensor<66x5x31xi32>, %arg1: tensor<66x31x26xi32>, %arg2: tensor<96x62x1x73xf32>, %arg3: tensor<52x32x90x50xf32>, %arg4: tensor<52xf32>, %arg5: tensor<42x69x31x84x53x93xi1>, %arg6: tensor<1x1x1x1x1x93xi1>) -> (tensor<42x69x31x84x53x93xi1>, tensor<8580xi32>, tensor<96x285x279x104xi1>, tensor<96x285x279x104xi1>, tensor<96x95x93x52xi1>, tensor<96x95x93x52xf32>, tensor<66x5x26xi32>, tensor<1x1x279x104xi1>, tensor<96x95x93x52xf32>, tensor<96x855x558x1xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<66x5x31xi32>, tensor<66x31x26xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<66x5x26xi32>
    %r_1 = tosa.const_shape {values = dense<[ 8580 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.reshape %0, %r_1 : (tensor<66x5x26xi32>, !tosa.shape<1>) -> tensor<8580xi32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 96, 95, 93, 52>} : (tensor<96x62x1x73xf32>, tensor<52x32x90x50xf32>, tensor<52xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<96x95x93x52xf32>
    %3 = tosa.logical_and %arg5, %arg6 : (tensor<42x69x31x84x53x93xi1>, tensor<1x1x1x1x1x93xi1>) -> tensor<42x69x31x84x53x93xi1>
    %4 = tosa.minimum %1, %1 : (tensor<8580xi32>, tensor<8580xi32>) -> tensor<8580xi32>
    %5 = tosa.greater_equal %2, %2 : (tensor<96x95x93x52xf32>, tensor<96x95x93x52xf32>) -> tensor<96x95x93x52xi1>
    %t_6 = tosa.const_shape {values = dense<[ 1, 3, 3, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %6 = tosa.tile %5, %t_6 : (tensor<96x95x93x52xi1>, !tosa.shape<4>) -> tensor<96x285x279x104xi1>
    %7 = tosa.bitwise_xor %6, %6 : (tensor<96x285x279x104xi1>, tensor<96x285x279x104xi1>) -> tensor<96x285x279x104xi1>
    %8 = tosa.sigmoid %2 : (tensor<96x95x93x52xf32>) -> tensor<96x95x93x52xf32>
    %9 = tosa.exp %2 : (tensor<96x95x93x52xf32>) -> tensor<96x95x93x52xf32>
    %10 = tosa.logical_and %7, %6 : (tensor<96x285x279x104xi1>, tensor<96x285x279x104xi1>) -> tensor<96x285x279x104xi1>
    %11 = tosa.sub %6, %6 : (tensor<96x285x279x104xi1>, tensor<96x285x279x104xi1>) -> tensor<96x285x279x104xi1>
    %12 = tosa.logical_right_shift %6, %11 : (tensor<96x285x279x104xi1>, tensor<96x285x279x104xi1>) -> tensor<96x285x279x104xi1>
    %13 = tosa.sub %6, %12 : (tensor<96x285x279x104xi1>, tensor<96x285x279x104xi1>) -> tensor<96x285x279x104xi1>
    %14 = tosa.logical_xor %6, %12 : (tensor<96x285x279x104xi1>, tensor<96x285x279x104xi1>) -> tensor<96x285x279x104xi1>
    %15 = tosa.greater_equal %2, %9 : (tensor<96x95x93x52xf32>, tensor<96x95x93x52xf32>) -> tensor<96x95x93x52xi1>
    %t_16 = tosa.const_shape {values = dense<[ 1, 3, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %16 = tosa.tile %10, %t_16 : (tensor<96x285x279x104xi1>, !tosa.shape<4>) -> tensor<96x855x558x104xi1>
    %17 = tosa.reduce_min %6 {axis = 1 : i32} : (tensor<96x285x279x104xi1>) -> tensor<96x1x279x104xi1>
    %18 = tosa.minimum %9, %8 : (tensor<96x95x93x52xf32>, tensor<96x95x93x52xf32>) -> tensor<96x95x93x52xf32>
    %19 = tosa.reduce_product %16 {axis = 3 : i32} : (tensor<96x855x558x104xi1>) -> tensor<96x855x558x1xi1>
    %20 = tosa.reduce_product %17 {axis = 0 : i32} : (tensor<96x1x279x104xi1>) -> tensor<1x1x279x104xi1>
    %21 = tosa.reduce_all %20 {axis = 1 : i32} : (tensor<1x1x279x104xi1>) -> tensor<1x1x279x104xi1>
    %22 = tosa.minimum %0, %0 : (tensor<66x5x26xi32>, tensor<66x5x26xi32>) -> tensor<66x5x26xi32>
    %23 = tosa.reciprocal %2 : (tensor<96x95x93x52xf32>) -> tensor<96x95x93x52xf32>
    %24 = tosa.bitwise_or %19, %19 : (tensor<96x855x558x1xi1>, tensor<96x855x558x1xi1>) -> tensor<96x855x558x1xi1>
    %25 = tosa.logical_xor %21, %20 : (tensor<1x1x279x104xi1>, tensor<1x1x279x104xi1>) -> tensor<1x1x279x104xi1>
    %26 = tosa.floor %23 : (tensor<96x95x93x52xf32>) -> tensor<96x95x93x52xf32>
    %27 = tosa.clz %24 : (tensor<96x855x558x1xi1>) -> tensor<96x855x558x1xi1>
    return %3, %4, %13, %14, %15, %18, %22, %25, %26, %27 : tensor<42x69x31x84x53x93xi1>, tensor<8580xi32>, tensor<96x285x279x104xi1>, tensor<96x285x279x104xi1>, tensor<96x95x93x52xi1>, tensor<96x95x93x52xf32>, tensor<66x5x26xi32>, tensor<1x1x279x104xi1>, tensor<96x95x93x52xf32>, tensor<96x855x558x1xi1>
  }
}
