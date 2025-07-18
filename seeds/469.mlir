module {
  func.func @main(%arg0: tensor<42x40xi64>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<1x2xi64>, tensor<1x1xi64>, tensor<1x1xi64>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<42x40xi64>) -> tensor<1x40xi64>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<1x40xi64>) -> tensor<1x40xi64>
    %2 = tosa.sub %1, %1 : (tensor<1x40xi64>, tensor<1x40xi64>) -> tensor<1x40xi64>
    %3 = tosa.reduce_max %2 {axis = 1 : i32} : (tensor<1x40xi64>) -> tensor<1x1xi64>
    %4 = tosa.maximum %3, %3 : (tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x1xi64>
    %5 = tosa.rsqrt %arg1 : (tensor<f32>) -> tensor<f32>
    %s_6_start = tosa.const_shape {values = dense<[ 0, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_6_size = tosa.const_shape {values = dense<[ 2, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.slice %4, %s_6_start, %s_6_size : (tensor<1x1xi64>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x2xi64>
    %7 = tosa.logical_left_shift %6, %6 : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
    %8 = tosa.reduce_min %7 {axis = 0 : i32} : (tensor<2x2xi64>) -> tensor<1x2xi64>
    %9 = tosa.tanh %5 : (tensor<f32>) -> tensor<f32>
    %10 = tosa.reciprocal %5 : (tensor<f32>) -> tensor<f32>
    %11 = tosa.minimum %8, %8 : (tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<1x2xi64>
    %12 = tosa.bitwise_and %4, %3 : (tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x1xi64>
    %13 = tosa.reduce_product %4 {axis = 0 : i32} : (tensor<1x1xi64>) -> tensor<1x1xi64>
    return %9, %10, %11, %12, %13 : tensor<f32>, tensor<f32>, tensor<1x2xi64>, tensor<1x1xi64>, tensor<1x1xi64>
  }
}
