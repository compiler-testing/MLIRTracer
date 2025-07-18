module {
  func.func @main(%arg0: tensor<10x2xf32>, %arg1: tensor<29x67x25x93xi8>, %arg2: tensor<1x1x1x1xi8>) -> (tensor<10x2xf32>, tensor<6x10x1x7xi1>, tensor<1x67x25x93xi1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<10x2xf32>) -> tensor<10x2xf32>
    %1 = tosa.add %0, %0 : (tensor<10x2xf32>, tensor<10x2xf32>) -> tensor<10x2xf32>
    %2 = tosa.logical_left_shift %arg1, %arg2 : (tensor<29x67x25x93xi8>, tensor<1x1x1x1xi8>) -> tensor<29x67x25x93xi8>
    %3 = tosa.equal %2, %2 : (tensor<29x67x25x93xi8>, tensor<29x67x25x93xi8>) -> tensor<29x67x25x93xi1>
    %4 = tosa.logical_left_shift %3, %3 : (tensor<29x67x25x93xi1>, tensor<29x67x25x93xi1>) -> tensor<29x67x25x93xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %3, %in_zp_5, %out_zp_5 : (tensor<29x67x25x93xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<29x67x25x93xi1>
    %6 = tosa.reciprocal %1 : (tensor<10x2xf32>) -> tensor<10x2xf32>
    %7 = tosa.tanh %6 : (tensor<10x2xf32>) -> tensor<10x2xf32>
    %s_8_start = tosa.const_shape {values = dense<[ 14, 24, 5, 22 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_8_size = tosa.const_shape {values = dense<[ 6, 10, 12, 7 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %8 = tosa.slice %4, %s_8_start, %s_8_size : (tensor<29x67x25x93xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<6x10x12x7xi1>
    %9 = tosa.logical_right_shift %5, %5 : (tensor<29x67x25x93xi1>, tensor<29x67x25x93xi1>) -> tensor<29x67x25x93xi1>
    %10 = tosa.logical_right_shift %8, %8 : (tensor<6x10x12x7xi1>, tensor<6x10x12x7xi1>) -> tensor<6x10x12x7xi1>
    %11 = tosa.pow %7, %6 : (tensor<10x2xf32>, tensor<10x2xf32>) -> tensor<10x2xf32>
    %12 = tosa.bitwise_or %10, %10 : (tensor<6x10x12x7xi1>, tensor<6x10x12x7xi1>) -> tensor<6x10x12x7xi1>
    %13 = tosa.rsqrt %11 : (tensor<10x2xf32>) -> tensor<10x2xf32>
    %14 = tosa.reduce_min %12 {axis = 2 : i32} : (tensor<6x10x12x7xi1>) -> tensor<6x10x1x7xi1>
    %15 = tosa.reduce_max %14 {axis = 2 : i32} : (tensor<6x10x1x7xi1>) -> tensor<6x10x1x7xi1>
    %16 = tosa.reduce_max %9 {axis = 0 : i32} : (tensor<29x67x25x93xi1>) -> tensor<1x67x25x93xi1>
    return %13, %15, %16 : tensor<10x2xf32>, tensor<6x10x1x7xi1>, tensor<1x67x25x93xi1>
  }
}
