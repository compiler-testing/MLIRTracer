module {
  func.func @main(%arg0: tensor<5x8x2x10x34xi64>, %arg1: tensor<42x42x52x19x65xf32>, %arg2: tensor<45xi64>, %arg3: tensor<81x41x85xi1>) -> (tensor<5x8x2x10x34xi1>, tensor<45xi64>, tensor<42x42x52x19x65xf32>, tensor<81x1x1xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<5x8x2x10x34xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<5x8x2x10x34xi64>
    %1 = tosa.bitwise_or %0, %0 : (tensor<5x8x2x10x34xi64>, tensor<5x8x2x10x34xi64>) -> tensor<5x8x2x10x34xi64>
    %2 = tosa.sigmoid %arg1 : (tensor<42x42x52x19x65xf32>) -> tensor<42x42x52x19x65xf32>
    %3 = tosa.greater_equal %1, %0 : (tensor<5x8x2x10x34xi64>, tensor<5x8x2x10x34xi64>) -> tensor<5x8x2x10x34xi1>
    %4 = tosa.bitwise_not %3 : (tensor<5x8x2x10x34xi1>) -> tensor<5x8x2x10x34xi1>
    %5 = tosa.reverse %arg2 {axis = 0 : i32} : (tensor<45xi64>) -> tensor<45xi64>
    %6 = tosa.reduce_all %arg3 {axis = 2 : i32} : (tensor<81x41x85xi1>) -> tensor<81x41x1xi1>
    %7 = tosa.rsqrt %2 : (tensor<42x42x52x19x65xf32>) -> tensor<42x42x52x19x65xf32>
    %8 = tosa.abs %7 : (tensor<42x42x52x19x65xf32>) -> tensor<42x42x52x19x65xf32>
    %9 = tosa.reduce_all %6 {axis = 1 : i32} : (tensor<81x41x1xi1>) -> tensor<81x1x1xi1>
    return %4, %5, %8, %9 : tensor<5x8x2x10x34xi1>, tensor<45xi64>, tensor<42x42x52x19x65xf32>, tensor<81x1x1xi1>
  }
}
