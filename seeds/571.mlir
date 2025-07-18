module {
  func.func @main(%arg0: tensor<80x80x36xi32>, %arg1: tensor<1x80x36xi32>, %arg2: tensor<64x90x26x6x85xi1>, %arg3: tensor<60x65x16x97x28xf32>) -> (tensor<60x65x16x97x28xf32>, tensor<60x65x16x97x28xf32>, tensor<80x240x36xi32>, tensor<1x240x36xi32>, tensor<60x65x16x97x28xf32>, tensor<64x90x26x6x85xi1>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<80x80x36xi32>, tensor<1x80x36xi32>) -> tensor<80x80x36xi32>
    %1 = tosa.logical_not %arg2 : (tensor<64x90x26x6x85xi1>) -> tensor<64x90x26x6x85xi1>
    %2 = tosa.sub %0, %0 : (tensor<80x80x36xi32>, tensor<80x80x36xi32>) -> tensor<80x80x36xi32>
    %3 = tosa.exp %arg3 : (tensor<60x65x16x97x28xf32>) -> tensor<60x65x16x97x28xf32>
    %4 = tosa.ceil %3 : (tensor<60x65x16x97x28xf32>) -> tensor<60x65x16x97x28xf32>
    %in_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<60x65x16x97x28xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<60x65x16x97x28xf32>
    %t_6 = tosa.const_shape {values = dense<[ 1, 3, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.tile %2, %t_6 : (tensor<80x80x36xi32>, !tosa.shape<3>) -> tensor<80x240x36xi32>
    %7 = tosa.reciprocal %5 : (tensor<60x65x16x97x28xf32>) -> tensor<60x65x16x97x28xf32>
    %8 = tosa.clamp %3 {min_val = 4.000000e+01 : f32, max_val = 1.530000e+02 : f32} : (tensor<60x65x16x97x28xf32>) -> tensor<60x65x16x97x28xf32>
    %9 = tosa.logical_left_shift %6, %6 : (tensor<80x240x36xi32>, tensor<80x240x36xi32>) -> tensor<80x240x36xi32>
    %10 = tosa.logical_left_shift %9, %6 : (tensor<80x240x36xi32>, tensor<80x240x36xi32>) -> tensor<80x240x36xi32>
    %11 = tosa.bitwise_and %6, %10 : (tensor<80x240x36xi32>, tensor<80x240x36xi32>) -> tensor<80x240x36xi32>
    %12 = tosa.reduce_product %9 {axis = 0 : i32} : (tensor<80x240x36xi32>) -> tensor<1x240x36xi32>
    %in_zp_13 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_13 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %13 = tosa.negate %3, %in_zp_13, %out_zp_13 : (tensor<60x65x16x97x28xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<60x65x16x97x28xf32>
    %14 = tosa.logical_xor %1, %1 : (tensor<64x90x26x6x85xi1>, tensor<64x90x26x6x85xi1>) -> tensor<64x90x26x6x85xi1>
    return %7, %8, %11, %12, %13, %14 : tensor<60x65x16x97x28xf32>, tensor<60x65x16x97x28xf32>, tensor<80x240x36xi32>, tensor<1x240x36xi32>, tensor<60x65x16x97x28xf32>, tensor<64x90x26x6x85xi1>
  }
}
