module {
  func.func @main(%arg0: tensor<37xi64>, %arg1: tensor<f32>, %arg2: tensor<30x62xi1>) -> (tensor<i32>, tensor<7xi64>, tensor<i32>, tensor<30x1xi1>, tensor<f32>, tensor<30x1xi1>, tensor<f32>, tensor<f32>, tensor<1x1xi1>, tensor<f32>, tensor<f32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 5 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_0_size = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<37xi64>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<8xi64>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<8xi64>) -> tensor<i32>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<8xi64>, tensor<8xi64>) -> tensor<16xi64>
    %s_4_start = tosa.const_shape {values = dense<[ 9 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_4_size = tosa.const_shape {values = dense<[ 7 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.slice %3, %s_4_start, %s_4_size : (tensor<16xi64>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<7xi64>
    %5 = tosa.reciprocal %arg1 : (tensor<f32>) -> tensor<f32>
    %6 = tosa.reduce_all %arg2 {axis = 1 : i32} : (tensor<30x62xi1>) -> tensor<30x1xi1>
    %7 = tosa.logical_or %6, %6 : (tensor<30x1xi1>, tensor<30x1xi1>) -> tensor<30x1xi1>
    %8 = tosa.intdiv %1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %9 = tosa.logical_and %7, %6 : (tensor<30x1xi1>, tensor<30x1xi1>) -> tensor<30x1xi1>
    %10 = tosa.reduce_max %6 {axis = 1 : i32} : (tensor<30x1xi1>) -> tensor<30x1xi1>
    %11 = tosa.logical_xor %10, %7 : (tensor<30x1xi1>, tensor<30x1xi1>) -> tensor<30x1xi1>
    %12 = tosa.intdiv %1, %8 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %13 = tosa.bitwise_or %10, %6 : (tensor<30x1xi1>, tensor<30x1xi1>) -> tensor<30x1xi1>
    %14 = tosa.reduce_any %10 {axis = 1 : i32} : (tensor<30x1xi1>) -> tensor<30x1xi1>
    %15 = tosa.floor %5 : (tensor<f32>) -> tensor<f32>
    %16 = tosa.logical_or %9, %11 : (tensor<30x1xi1>, tensor<30x1xi1>) -> tensor<30x1xi1>
    %17 = tosa.reduce_all %14 {axis = 0 : i32} : (tensor<30x1xi1>) -> tensor<1x1xi1>
    %18 = tosa.reduce_product %17 {axis = 0 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %19 = tosa.floor %5 : (tensor<f32>) -> tensor<f32>
    %20 = tosa.exp %5 : (tensor<f32>) -> tensor<f32>
    %21 = tosa.logical_or %17, %18 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %22 = tosa.ceil %5 : (tensor<f32>) -> tensor<f32>
    %23 = tosa.floor %5 : (tensor<f32>) -> tensor<f32>
    return %2, %4, %12, %13, %15, %16, %19, %20, %21, %22, %23 : tensor<i32>, tensor<7xi64>, tensor<i32>, tensor<30x1xi1>, tensor<f32>, tensor<30x1xi1>, tensor<f32>, tensor<f32>, tensor<1x1xi1>, tensor<f32>, tensor<f32>
  }
}
