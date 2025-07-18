module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<i16>, %arg2: tensor<63x46xf32>) -> (tensor<i16>, tensor<63x46xi1>, tensor<63x1xf32>, tensor<6x2xi1>, tensor<6x2xi1>, tensor<1x1xf32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %1 = tosa.ceil %arg2 : (tensor<63x46xf32>) -> tensor<63x46xf32>
    %2 = tosa.equal %1, %1 : (tensor<63x46xf32>, tensor<63x46xf32>) -> tensor<63x46xi1>
    %3 = tosa.floor %1 : (tensor<63x46xf32>) -> tensor<63x46xf32>
    %4 = tosa.add %1, %3 : (tensor<63x46xf32>, tensor<63x46xf32>) -> tensor<63x46xf32>
    %s_5_start = tosa.const_shape {values = dense<[ 8, 44 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_5_size = tosa.const_shape {values = dense<[ 6, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.slice %3, %s_5_start, %s_5_size : (tensor<63x46xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<6x2xf32>
    %6 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %7 = tosa.transpose %5 {perms = array<i32: 0, 1>} : (tensor<6x2xf32>) -> tensor<6x2xf32>
    %8 = tosa.tanh %4 : (tensor<63x46xf32>) -> tensor<63x46xf32>
    %9 = tosa.reverse %8 {axis = 1 : i32} : (tensor<63x46xf32>) -> tensor<63x46xf32>
    %10 = tosa.greater %7, %5 : (tensor<6x2xf32>, tensor<6x2xf32>) -> tensor<6x2xi1>
    %11 = tosa.tanh %9 : (tensor<63x46xf32>) -> tensor<63x46xf32>
    %s_12_start = tosa.const_shape {values = dense<[ 25, 24 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_12_size = tosa.const_shape {values = dense<[ 12, 4 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %12 = tosa.slice %11, %s_12_start, %s_12_size : (tensor<63x46xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<12x4xf32>
    %13 = tosa.concat %12, %12 {axis = 1 : i32} : (tensor<12x4xf32>, tensor<12x4xf32>) -> tensor<12x8xf32>
    %t_14 = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %14 = tosa.tile %9, %t_14 : (tensor<63x46xf32>, !tosa.shape<2>) -> tensor<63x92xf32>
    %15 = tosa.reduce_sum %14 {axis = 1 : i32} : (tensor<63x92xf32>) -> tensor<63x1xf32>
    %16 = tosa.clamp %13 {min_val = 3.600000e+01 : f32, max_val = 9.200000e+01 : f32} : (tensor<12x8xf32>) -> tensor<12x8xf32>
    %17 = tosa.bitwise_not %10 : (tensor<6x2xi1>) -> tensor<6x2xi1>
    %t_18 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %18 = tosa.tile %16, %t_18 : (tensor<12x8xf32>, !tosa.shape<2>) -> tensor<36x16xf32>
    %19 = tosa.logical_not %10 : (tensor<6x2xi1>) -> tensor<6x2xi1>
    %20 = tosa.reduce_max %18 {axis = 1 : i32} : (tensor<36x16xf32>) -> tensor<36x1xf32>
    %21 = tosa.reduce_sum %20 {axis = 0 : i32} : (tensor<36x1xf32>) -> tensor<1x1xf32>
    return %0, %2, %15, %17, %19, %21 : tensor<i16>, tensor<63x46xi1>, tensor<63x1xf32>, tensor<6x2xi1>, tensor<6x2xi1>, tensor<1x1xf32>
  }
}
