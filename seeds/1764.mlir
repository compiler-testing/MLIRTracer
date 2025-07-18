module {
  func.func @main(%arg0: tensor<70x22x37x95xf32>, %arg1: tensor<25x83xi16>, %arg2: tensor<25x1xi16>, %arg3: tensor<i1>, %arg4: tensor<i1>, %arg5: tensor<100x39x8xi1>) -> (tensor<4x9x10x6xf32>, tensor<70x22x37x95xf32>, tensor<i1>, tensor<70x22x37x95xf32>, tensor<25x83xi16>, tensor<100x1x8xi1>, tensor<25x1xi16>, tensor<1x83xi16>) {
    %0 = tosa.log %arg0 : (tensor<70x22x37x95xf32>) -> tensor<70x22x37x95xf32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<25x83xi16>, tensor<25x1xi16>) -> tensor<25x83xi16>
    %s_2_start = tosa.const_shape {values = dense<[ 26, 5, 27, 11 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_2_size = tosa.const_shape {values = dense<[ 4, 9, 10, 6 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.slice %0, %s_2_start, %s_2_size : (tensor<70x22x37x95xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<4x9x10x6xf32>
    %3 = tosa.bitwise_not %1 : (tensor<25x83xi16>) -> tensor<25x83xi16>
    %4 = tosa.log %0 : (tensor<70x22x37x95xf32>) -> tensor<70x22x37x95xf32>
    %5 = tosa.reduce_sum %3 {axis = 0 : i32} : (tensor<25x83xi16>) -> tensor<1x83xi16>
    %6 = tosa.logical_or %arg3, %arg4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.sigmoid %0 : (tensor<70x22x37x95xf32>) -> tensor<70x22x37x95xf32>
    %8 = tosa.abs %1 : (tensor<25x83xi16>) -> tensor<25x83xi16>
    %9 = tosa.reduce_all %arg5 {axis = 1 : i32} : (tensor<100x39x8xi1>) -> tensor<100x1x8xi1>
    %in_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %10 = tosa.negate %5, %in_zp_10, %out_zp_10 : (tensor<1x83xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<1x83xi16>
    %11 = tosa.clamp %10 {min_val = -57 : i16, max_val = -40 : i16} : (tensor<1x83xi16>) -> tensor<1x83xi16>
    %12 = tosa.reduce_max %1 {axis = 1 : i32} : (tensor<25x83xi16>) -> tensor<25x1xi16>
    %13 = tosa.arithmetic_right_shift %12, %12 {round = true} : (tensor<25x1xi16>, tensor<25x1xi16>) -> tensor<25x1xi16>
    %in_zp_14 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_14 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %14 = tosa.negate %11, %in_zp_14, %out_zp_14 : (tensor<1x83xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<1x83xi16>
    return %2, %4, %6, %7, %8, %9, %13, %14 : tensor<4x9x10x6xf32>, tensor<70x22x37x95xf32>, tensor<i1>, tensor<70x22x37x95xf32>, tensor<25x83xi16>, tensor<100x1x8xi1>, tensor<25x1xi16>, tensor<1x83xi16>
  }
}
