module {
  func.func @main(%arg0: tensor<42x68xf32>, %arg1: tensor<i16>, %arg2: tensor<i16>) -> (tensor<1x1xf32>, tensor<i16>, tensor<1x1xf32>, tensor<1x68xf32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<42x68xf32>) -> tensor<42x68xf32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %2 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<42x68xf32>) -> tensor<1x68xf32>
    %3 = tosa.reduce_max %2 {axis = 1 : i32} : (tensor<1x68xf32>) -> tensor<1x1xf32>
    %4 = tosa.exp %3 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %5 = tosa.minimum %4, %3 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %6 = tosa.reduce_sum %5 {axis = 1 : i32} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %7 = tosa.negate %1, %in_zp_7, %out_zp_7 : (tensor<i16>, tensor<1xi16>, tensor<1xi16>) -> tensor<i16>
    %8 = tosa.arithmetic_right_shift %7, %1 {round = false} : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %in_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %9 = tosa.negate %8, %in_zp_9, %out_zp_9 : (tensor<i16>, tensor<1xi16>, tensor<1xi16>) -> tensor<i16>
    %10 = tosa.sigmoid %5 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %11 = tosa.minimum %2, %2 : (tensor<1x68xf32>, tensor<1x68xf32>) -> tensor<1x68xf32>
    return %6, %9, %10, %11 : tensor<1x1xf32>, tensor<i16>, tensor<1x1xf32>, tensor<1x68xf32>
  }
}
