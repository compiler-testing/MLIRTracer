module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<50x90x21xi1>, %arg3: tensor<1x90x21xi1>, %arg4: tensor<60x46xi32>, %arg5: tensor<60x46xi32>) -> (tensor<f32>, tensor<60x46xi32>, tensor<50x1x21xi1>, tensor<60x1xi32>, tensor<60x46xi32>, tensor<f32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = tosa.exp %0 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<50x90x21xi1>, tensor<1x90x21xi1>) -> tensor<50x90x21xi1>
    %3 = tosa.logical_right_shift %2, %2 : (tensor<50x90x21xi1>, tensor<50x90x21xi1>) -> tensor<50x90x21xi1>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %4 = tosa.negate %3, %in_zp_4, %out_zp_4 : (tensor<50x90x21xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<50x90x21xi1>
    %5 = tosa.minimum %arg4, %arg5 : (tensor<60x46xi32>, tensor<60x46xi32>) -> tensor<60x46xi32>
    %6 = tosa.maximum %5, %5 : (tensor<60x46xi32>, tensor<60x46xi32>) -> tensor<60x46xi32>
    %7 = tosa.log %1 : (tensor<f32>) -> tensor<f32>
    %8 = tosa.sub %5, %6 : (tensor<60x46xi32>, tensor<60x46xi32>) -> tensor<60x46xi32>
    %9 = tosa.reverse %6 {axis = 0 : i32} : (tensor<60x46xi32>) -> tensor<60x46xi32>
    %10 = tosa.clz %5 : (tensor<60x46xi32>) -> tensor<60x46xi32>
    %11 = tosa.reduce_any %4 {axis = 1 : i32} : (tensor<50x90x21xi1>) -> tensor<50x1x21xi1>
    %12 = tosa.reduce_sum %10 {axis = 1 : i32} : (tensor<60x46xi32>) -> tensor<60x1xi32>
    %13 = tosa.intdiv %9, %9 : (tensor<60x46xi32>, tensor<60x46xi32>) -> tensor<60x46xi32>
    %14 = tosa.rsqrt %1 : (tensor<f32>) -> tensor<f32>
    return %7, %8, %11, %12, %13, %14 : tensor<f32>, tensor<60x46xi32>, tensor<50x1x21xi1>, tensor<60x1xi32>, tensor<60x46xi32>, tensor<f32>
  }
}
