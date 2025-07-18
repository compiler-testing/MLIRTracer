module {
  func.func @main(%arg0: tensor<26x17xi64>, %arg1: tensor<1x1xi64>, %arg2: tensor<91x71x36x19xf32>, %arg3: tensor<65x36x12xi1>) -> (tensor<91x19x71x36xf32>, tensor<26x17xi64>, tensor<65x36x1xi1>, tensor<91x71x36x19xf32>, tensor<91x71x36x19xf32>, tensor<65x36xi32>, tensor<91x71x36x19xf32>, tensor<91x71x36x19xf32>, tensor<65x36x1xi1>, tensor<65x36x1xi1>, tensor<91x71x36x19xf32>, tensor<91x71x36x19xf32>, tensor<3x8xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<26x17xi64>, tensor<1x1xi64>) -> tensor<26x17xi64>
    %s_1_start = tosa.const_shape {values = dense<[ 23, 9 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_1_size = tosa.const_shape {values = dense<[ 3, 8 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<26x17xi64>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x8xi64>
    %2 = tosa.rsqrt %arg2 : (tensor<91x71x36x19xf32>) -> tensor<91x71x36x19xf32>
    %3 = tosa.tanh %2 : (tensor<91x71x36x19xf32>) -> tensor<91x71x36x19xf32>
    %4 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %5 = tosa.transpose %2 {perms = array<i32: 0, 3, 1, 2>} : (tensor<91x71x36x19xf32>) -> tensor<91x19x71x36xf32>
    %6 = tosa.logical_left_shift %0, %0 : (tensor<26x17xi64>, tensor<26x17xi64>) -> tensor<26x17xi64>
    %7 = tosa.bitwise_and %1, %1 : (tensor<3x8xi64>, tensor<3x8xi64>) -> tensor<3x8xi64>
    %8 = tosa.logical_right_shift %1, %7 : (tensor<3x8xi64>, tensor<3x8xi64>) -> tensor<3x8xi64>
    %9 = tosa.clamp %3 {min_val = 4.600000e+01 : f32, max_val = 9.300000e+01 : f32} : (tensor<91x71x36x19xf32>) -> tensor<91x71x36x19xf32>
    %10 = tosa.reduce_all %arg3 {axis = 2 : i32} : (tensor<65x36x12xi1>) -> tensor<65x36x1xi1>
    %11 = tosa.floor %9 : (tensor<91x71x36x19xf32>) -> tensor<91x71x36x19xf32>
    %12 = tosa.logical_not %10 : (tensor<65x36x1xi1>) -> tensor<65x36x1xi1>
    %13 = tosa.logical_not %10 : (tensor<65x36x1xi1>) -> tensor<65x36x1xi1>
    %14 = tosa.reduce_max %13 {axis = 2 : i32} : (tensor<65x36x1xi1>) -> tensor<65x36x1xi1>
    %15 = tosa.minimum %2, %3 : (tensor<91x71x36x19xf32>, tensor<91x71x36x19xf32>) -> tensor<91x71x36x19xf32>
    %16 = tosa.bitwise_and %12, %12 : (tensor<65x36x1xi1>, tensor<65x36x1xi1>) -> tensor<65x36x1xi1>
    %17 = tosa.exp %2 : (tensor<91x71x36x19xf32>) -> tensor<91x71x36x19xf32>
    %18 = tosa.argmax %13 {axis = 2 : i32} : (tensor<65x36x1xi1>) -> tensor<65x36xi32>
    %19 = tosa.reciprocal %2 : (tensor<91x71x36x19xf32>) -> tensor<91x71x36x19xf32>
    %20 = tosa.floor %3 : (tensor<91x71x36x19xf32>) -> tensor<91x71x36x19xf32>
    %21 = tosa.logical_right_shift %13, %12 : (tensor<65x36x1xi1>, tensor<65x36x1xi1>) -> tensor<65x36x1xi1>
    %22 = tosa.logical_not %16 : (tensor<65x36x1xi1>) -> tensor<65x36x1xi1>
    %23 = tosa.logical_left_shift %22, %12 : (tensor<65x36x1xi1>, tensor<65x36x1xi1>) -> tensor<65x36x1xi1>
    %24 = tosa.floor %11 : (tensor<91x71x36x19xf32>) -> tensor<91x71x36x19xf32>
    %25 = tosa.tanh %9 : (tensor<91x71x36x19xf32>) -> tensor<91x71x36x19xf32>
    %26 = tosa.equal %7, %8 : (tensor<3x8xi64>, tensor<3x8xi64>) -> tensor<3x8xi1>
    return %5, %6, %14, %15, %17, %18, %19, %20, %21, %23, %24, %25, %26 : tensor<91x19x71x36xf32>, tensor<26x17xi64>, tensor<65x36x1xi1>, tensor<91x71x36x19xf32>, tensor<91x71x36x19xf32>, tensor<65x36xi32>, tensor<91x71x36x19xf32>, tensor<91x71x36x19xf32>, tensor<65x36x1xi1>, tensor<65x36x1xi1>, tensor<91x71x36x19xf32>, tensor<91x71x36x19xf32>, tensor<3x8xi1>
  }
}
