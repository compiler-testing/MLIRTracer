module {
  func.func @main(%arg0: tensor<8x69x42xi1>, %arg1: tensor<1x69x1xi1>, %arg2: tensor<56x20x39x35xf32>) -> (tensor<6x2x7x3xf32>, tensor<56x20x39x1xf32>, tensor<8x69x42xi1>, tensor<8x1x42xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<8x69x42xi1>, tensor<1x69x1xi1>) -> tensor<8x69x42xi1>
    %1 = tosa.floor %arg2 : (tensor<56x20x39x35xf32>) -> tensor<56x20x39x35xf32>
    %2 = tosa.bitwise_or %0, %0 : (tensor<8x69x42xi1>, tensor<8x69x42xi1>) -> tensor<8x69x42xi1>
    %3 = tosa.sub %2, %0 : (tensor<8x69x42xi1>, tensor<8x69x42xi1>) -> tensor<8x69x42xi1>
    %4 = tosa.clz %3 : (tensor<8x69x42xi1>) -> tensor<8x69x42xi1>
    %5 = tosa.clz %4 : (tensor<8x69x42xi1>) -> tensor<8x69x42xi1>
    %6 = tosa.reduce_product %1 {axis = 3 : i32} : (tensor<56x20x39x35xf32>) -> tensor<56x20x39x1xf32>
    %7 = tosa.rsqrt %6 : (tensor<56x20x39x1xf32>) -> tensor<56x20x39x1xf32>
    %s_8_start = tosa.const_shape {values = dense<[ 43, 4, 5, 0 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_8_size = tosa.const_shape {values = dense<[ 6, 2, 7, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %8 = tosa.slice %6, %s_8_start, %s_8_size : (tensor<56x20x39x1xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<6x2x7x3xf32>
    %9 = tosa.logical_right_shift %0, %5 : (tensor<8x69x42xi1>, tensor<8x69x42xi1>) -> tensor<8x69x42xi1>
    %10 = tosa.ceil %8 : (tensor<6x2x7x3xf32>) -> tensor<6x2x7x3xf32>
    %in_zp_11 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_11 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %11 = tosa.negate %7, %in_zp_11, %out_zp_11 : (tensor<56x20x39x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<56x20x39x1xf32>
    %12 = tosa.logical_xor %5, %9 : (tensor<8x69x42xi1>, tensor<8x69x42xi1>) -> tensor<8x69x42xi1>
    %13 = tosa.bitwise_and %12, %12 : (tensor<8x69x42xi1>, tensor<8x69x42xi1>) -> tensor<8x69x42xi1>
    %14 = tosa.reduce_any %4 {axis = 1 : i32} : (tensor<8x69x42xi1>) -> tensor<8x1x42xi1>
    return %10, %11, %13, %14 : tensor<6x2x7x3xf32>, tensor<56x20x39x1xf32>, tensor<8x69x42xi1>, tensor<8x1x42xi1>
  }
}
