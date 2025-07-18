module {
  func.func @main(%arg0: tensor<36x49x57x85xi32>, %arg1: tensor<36x1x57x85xi32>, %arg2: tensor<30x14xf32>, %arg3: tensor<37xi1>, %arg4: tensor<37xi1>) -> (tensor<36x49x57x85xi32>, tensor<1x14xf32>, tensor<2x14xi1>, tensor<30x14xf32>, tensor<37xi1>, tensor<37xi1>, tensor<37xi1>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<36x49x57x85xi32>, tensor<36x1x57x85xi32>) -> tensor<36x49x57x85xi32>
    %1 = tosa.sigmoid %arg2 : (tensor<30x14xf32>) -> tensor<30x14xf32>
    %2 = tosa.floor %1 : (tensor<30x14xf32>) -> tensor<30x14xf32>
    %3 = tosa.maximum %1, %1 : (tensor<30x14xf32>, tensor<30x14xf32>) -> tensor<30x14xf32>
    %4 = tosa.reduce_sum %3 {axis = 0 : i32} : (tensor<30x14xf32>) -> tensor<1x14xf32>
    %5 = tosa.clamp %4 {min_val = 1.700000e+01 : f32, max_val = 8.700000e+01 : f32} : (tensor<1x14xf32>) -> tensor<1x14xf32>
    %6 = tosa.maximum %2, %2 : (tensor<30x14xf32>, tensor<30x14xf32>) -> tensor<30x14xf32>
    %7 = tosa.logical_and %arg3, %arg4 : (tensor<37xi1>, tensor<37xi1>) -> tensor<37xi1>
    %in_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %8 = tosa.negate %6, %in_zp_8, %out_zp_8 : (tensor<30x14xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<30x14xf32>
    %9 = tosa.intdiv %0, %0 : (tensor<36x49x57x85xi32>, tensor<36x49x57x85xi32>) -> tensor<36x49x57x85xi32>
    %10 = tosa.log %5 : (tensor<1x14xf32>) -> tensor<1x14xf32>
    %11 = tosa.sub %7, %7 : (tensor<37xi1>, tensor<37xi1>) -> tensor<37xi1>
    %12 = tosa.greater_equal %8, %1 : (tensor<30x14xf32>, tensor<30x14xf32>) -> tensor<30x14xi1>
    %13 = tosa.pow %5, %10 : (tensor<1x14xf32>, tensor<1x14xf32>) -> tensor<1x14xf32>
    %14 = tosa.logical_not %7 : (tensor<37xi1>) -> tensor<37xi1>
    %15 = tosa.reduce_max %12 {axis = 0 : i32} : (tensor<30x14xi1>) -> tensor<1x14xi1>
    %16 = tosa.logical_right_shift %14, %11 : (tensor<37xi1>, tensor<37xi1>) -> tensor<37xi1>
    %17 = tosa.logical_right_shift %15, %15 : (tensor<1x14xi1>, tensor<1x14xi1>) -> tensor<1x14xi1>
    %t_18 = tosa.const_shape {values = dense<[ 2, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %18 = tosa.tile %17, %t_18 : (tensor<1x14xi1>, !tosa.shape<2>) -> tensor<2x14xi1>
    %19 = tosa.pow %8, %2 : (tensor<30x14xf32>, tensor<30x14xf32>) -> tensor<30x14xf32>
    %20 = tosa.bitwise_xor %16, %7 : (tensor<37xi1>, tensor<37xi1>) -> tensor<37xi1>
    %21 = tosa.bitwise_xor %16, %11 : (tensor<37xi1>, tensor<37xi1>) -> tensor<37xi1>
    %22 = tosa.bitwise_not %16 : (tensor<37xi1>) -> tensor<37xi1>
    return %9, %13, %18, %19, %20, %21, %22 : tensor<36x49x57x85xi32>, tensor<1x14xf32>, tensor<2x14xi1>, tensor<30x14xf32>, tensor<37xi1>, tensor<37xi1>, tensor<37xi1>
  }
}
