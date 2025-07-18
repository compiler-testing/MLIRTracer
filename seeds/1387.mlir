module {
  func.func @main(%arg0: tensor<41x98x17x66x90xi32>, %arg1: tensor<1x98x17x66x1xi32>, %arg2: tensor<74xi1>, %arg3: tensor<74xi1>, %arg4: tensor<24x73x4x89xf32>, %arg5: tensor<30x47x8x44xf32>, %arg6: tensor<30xf32>) -> (tensor<74xi1>, tensor<1x123x18x30xf32>, tensor<41x98x17x66x90xi32>, tensor<24x123x18x1xi1>, tensor<1x123x18x30xf32>, tensor<1x123x18x30xf32>, tensor<10x7x4x8xf32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<41x98x17x66x90xi32>, tensor<1x98x17x66x1xi32>) -> tensor<41x98x17x66x90xi32>
    %1 = tosa.logical_and %arg2, %arg3 : (tensor<74xi1>, tensor<74xi1>) -> tensor<74xi1>
    %t_2 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.tile %1, %t_2 : (tensor<74xi1>, !tosa.shape<1>) -> tensor<74xi1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg4, %arg5, %arg6, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 24, 123, 18, 30>} : (tensor<24x73x4x89xf32>, tensor<30x47x8x44xf32>, tensor<30xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<24x123x18x30xf32>
    %4 = tosa.greater %3, %3 : (tensor<24x123x18x30xf32>, tensor<24x123x18x30xf32>) -> tensor<24x123x18x30xi1>
    %5 = tosa.reduce_min %3 {axis = 0 : i32} : (tensor<24x123x18x30xf32>) -> tensor<1x123x18x30xf32>
    %6 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<24x123x18x30xf32>) -> tensor<1x123x18x30xf32>
    %7 = tosa.abs %4 : (tensor<24x123x18x30xi1>) -> tensor<24x123x18x30xi1>
    %8 = tosa.add %6, %5 : (tensor<1x123x18x30xf32>, tensor<1x123x18x30xf32>) -> tensor<1x123x18x30xf32>
    %9 = tosa.minimum %6, %6 : (tensor<1x123x18x30xf32>, tensor<1x123x18x30xf32>) -> tensor<1x123x18x30xf32>
    %10 = tosa.intdiv %0, %0 : (tensor<41x98x17x66x90xi32>, tensor<41x98x17x66x90xi32>) -> tensor<41x98x17x66x90xi32>
    %11 = tosa.reduce_sum %9 {axis = 0 : i32} : (tensor<1x123x18x30xf32>) -> tensor<1x123x18x30xf32>
    %12 = tosa.logical_left_shift %7, %7 : (tensor<24x123x18x30xi1>, tensor<24x123x18x30xi1>) -> tensor<24x123x18x30xi1>
    %13 = tosa.reduce_max %12 {axis = 3 : i32} : (tensor<24x123x18x30xi1>) -> tensor<24x123x18x1xi1>
    %14 = tosa.pow %9, %11 : (tensor<1x123x18x30xf32>, tensor<1x123x18x30xf32>) -> tensor<1x123x18x30xf32>
    %15 = tosa.reciprocal %11 : (tensor<1x123x18x30xf32>) -> tensor<1x123x18x30xf32>
    %16 = tosa.rsqrt %11 : (tensor<1x123x18x30xf32>) -> tensor<1x123x18x30xf32>
    %s_17_start = tosa.const_shape {values = dense<[ 0, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_17_size = tosa.const_shape {values = dense<[ 10, 7, 4, 8 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %17 = tosa.slice %16, %s_17_start, %s_17_size : (tensor<1x123x18x30xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<10x7x4x8xf32>
    return %2, %8, %10, %13, %14, %15, %17 : tensor<74xi1>, tensor<1x123x18x30xf32>, tensor<41x98x17x66x90xi32>, tensor<24x123x18x1xi1>, tensor<1x123x18x30xf32>, tensor<1x123x18x30xf32>, tensor<10x7x4x8xf32>
  }
}
