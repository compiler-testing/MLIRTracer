module {
  func.func @main(%arg0: tensor<26x54x62xi1>, %arg1: tensor<1x54x62xi1>, %arg2: tensor<100x56xf32>) -> (tensor<1x54x62xi1>, tensor<26x54x62xi1>, tensor<1x1xi32>, tensor<26x1x62xi1>, tensor<100x56xf32>, tensor<26x1xi32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<26x54x62xi1>, tensor<1x54x62xi1>) -> tensor<26x54x62xi1>
    %1 = tosa.reduce_all %0 {axis = 1 : i32} : (tensor<26x54x62xi1>) -> tensor<26x1x62xi1>
    %2 = tosa.argmax %1 {axis = 2 : i32} : (tensor<26x1x62xi1>) -> tensor<26x1xi32>
    %3 = tosa.bitwise_or %2, %2 : (tensor<26x1xi32>, tensor<26x1xi32>) -> tensor<26x1xi32>
    %4 = tosa.minimum %3, %3 : (tensor<26x1xi32>, tensor<26x1xi32>) -> tensor<26x1xi32>
    %5 = tosa.reverse %4 {axis = 1 : i32} : (tensor<26x1xi32>) -> tensor<26x1xi32>
    %6 = tosa.reduce_product %5 {axis = 0 : i32} : (tensor<26x1xi32>) -> tensor<1x1xi32>
    %7 = tosa.reduce_all %0 {axis = 0 : i32} : (tensor<26x54x62xi1>) -> tensor<1x54x62xi1>
    %8 = tosa.clamp %6 {min_val = -61 : i32, max_val = 13 : i32} : (tensor<1x1xi32>) -> tensor<1x1xi32>
    %9 = tosa.logical_xor %0, %0 : (tensor<26x54x62xi1>, tensor<26x54x62xi1>) -> tensor<26x54x62xi1>
    %10 = tosa.clamp %8 {min_val = -61 : i32, max_val = 13 : i32} : (tensor<1x1xi32>) -> tensor<1x1xi32>
    %11 = tosa.sigmoid %arg2 : (tensor<100x56xf32>) -> tensor<100x56xf32>
    %12 = tosa.rsqrt %11 : (tensor<100x56xf32>) -> tensor<100x56xf32>
    %13 = tosa.maximum %12, %12 : (tensor<100x56xf32>, tensor<100x56xf32>) -> tensor<100x56xf32>
    %14 = tosa.logical_not %1 : (tensor<26x1x62xi1>) -> tensor<26x1x62xi1>
    %15 = tosa.clamp %13 {min_val = -1.800000e+01 : f32, max_val = 1.300000e+01 : f32} : (tensor<100x56xf32>) -> tensor<100x56xf32>
    %16 = tosa.bitwise_and %3, %2 : (tensor<26x1xi32>, tensor<26x1xi32>) -> tensor<26x1xi32>
    return %7, %9, %10, %14, %15, %16 : tensor<1x54x62xi1>, tensor<26x54x62xi1>, tensor<1x1xi32>, tensor<26x1x62xi1>, tensor<100x56xf32>, tensor<26x1xi32>
  }
}
