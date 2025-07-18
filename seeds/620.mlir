module {
  func.func @main(%arg0: tensor<75x86x8x30x23xi1>, %arg1: tensor<75x1x8x30x23xi1>, %arg2: tensor<73xf32>) -> (tensor<1xf32>, tensor<75x86x8x30x23xi1>, tensor<1xf32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<75x86x8x30x23xi1>, tensor<75x1x8x30x23xi1>) -> tensor<75x86x8x30x23xi1>
    %1 = tosa.sigmoid %arg2 : (tensor<73xf32>) -> tensor<73xf32>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<73xf32>) -> tensor<1xf32>
    %3 = tosa.clamp %1 {min_val = -5.600000e+01 : f32, max_val = 8.100000e+01 : f32} : (tensor<73xf32>) -> tensor<73xf32>
    %4 = tosa.bitwise_not %0 : (tensor<75x86x8x30x23xi1>) -> tensor<75x86x8x30x23xi1>
    %5 = tosa.logical_xor %0, %4 : (tensor<75x86x8x30x23xi1>, tensor<75x86x8x30x23xi1>) -> tensor<75x86x8x30x23xi1>
    %6 = tosa.reduce_max %3 {axis = 0 : i32} : (tensor<73xf32>) -> tensor<1xf32>
    return %2, %5, %6 : tensor<1xf32>, tensor<75x86x8x30x23xi1>, tensor<1xf32>
  }
}
