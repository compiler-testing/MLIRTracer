module {
  func.func @main(%arg0: tensor<6x21xf32>, %arg1: tensor<i16>, %arg2: tensor<26x99x75x14x66xi1>) -> (tensor<26x99x75x14x66xi1>, tensor<21xi32>, tensor<1x21xf32>, tensor<i16>, tensor<6x1xf32>, tensor<1x21xf32>, tensor<i16>, tensor<6x21xf32>, tensor<2x1xf32>, tensor<18x63xf32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<6x21xf32>) -> tensor<6x21xf32>
    %1 = tosa.bitwise_not %arg1 : (tensor<i16>) -> tensor<i16>
    %2 = tosa.logical_not %arg2 : (tensor<26x99x75x14x66xi1>) -> tensor<26x99x75x14x66xi1>
    %3 = tosa.clamp %1 {min_val = -46 : i16, max_val = -11 : i16} : (tensor<i16>) -> tensor<i16>
    %4 = tosa.maximum %0, %0 : (tensor<6x21xf32>, tensor<6x21xf32>) -> tensor<6x21xf32>
    %5 = tosa.bitwise_xor %3, %1 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %6 = tosa.logical_xor %2, %2 : (tensor<26x99x75x14x66xi1>, tensor<26x99x75x14x66xi1>) -> tensor<26x99x75x14x66xi1>
    %7 = tosa.clamp %5 {min_val = -46 : i16, max_val = -11 : i16} : (tensor<i16>) -> tensor<i16>
    %8 = tosa.reduce_sum %4 {axis = 0 : i32} : (tensor<6x21xf32>) -> tensor<1x21xf32>
    %9 = tosa.ceil %0 : (tensor<6x21xf32>) -> tensor<6x21xf32>
    %10 = tosa.bitwise_or %7, %3 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %11 = tosa.rsqrt %9 : (tensor<6x21xf32>) -> tensor<6x21xf32>
    %12 = tosa.argmax %8 {axis = 0 : i32} : (tensor<1x21xf32>) -> tensor<21xi32>
    %13 = tosa.sigmoid %0 : (tensor<6x21xf32>) -> tensor<6x21xf32>
    %14 = tosa.sigmoid %9 : (tensor<6x21xf32>) -> tensor<6x21xf32>
    %15 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<6x21xf32>) -> tensor<1x21xf32>
    %16 = tosa.bitwise_xor %3, %3 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %17 = tosa.reduce_min %11 {axis = 1 : i32} : (tensor<6x21xf32>) -> tensor<6x1xf32>
    %18 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<6x21xf32>) -> tensor<1x21xf32>
    %19 = tosa.bitwise_and %10, %10 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %20 = tosa.reciprocal %14 : (tensor<6x21xf32>) -> tensor<6x21xf32>
    %s_21_start = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_21_size = tosa.const_shape {values = dense<[ 2, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %21 = tosa.slice %11, %s_21_start, %s_21_size : (tensor<6x21xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x3xf32>
    %22 = tosa.reduce_max %21 {axis = 1 : i32} : (tensor<2x3xf32>) -> tensor<2x1xf32>
    %t_23 = tosa.const_shape {values = dense<[ 3, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %23 = tosa.tile %13, %t_23 : (tensor<6x21xf32>, !tosa.shape<2>) -> tensor<18x63xf32>
    return %6, %12, %15, %16, %17, %18, %19, %20, %22, %23 : tensor<26x99x75x14x66xi1>, tensor<21xi32>, tensor<1x21xf32>, tensor<i16>, tensor<6x1xf32>, tensor<1x21xf32>, tensor<i16>, tensor<6x21xf32>, tensor<2x1xf32>, tensor<18x63xf32>
  }
}
