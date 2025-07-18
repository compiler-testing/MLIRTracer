module {
  func.func @main(%arg0: tensor<74x30x80x71xi32>, %arg1: tensor<19x11x82xf32>) -> (tensor<74x1x80x71xi1>, tensor<19x11x82xf32>, tensor<1x11x82xf32>) {
    %0 = tosa.clz %arg0 : (tensor<74x30x80x71xi32>) -> tensor<74x30x80x71xi32>
    %1 = tosa.bitwise_and %0, %0 : (tensor<74x30x80x71xi32>, tensor<74x30x80x71xi32>) -> tensor<74x30x80x71xi32>
    %2 = tosa.intdiv %1, %0 : (tensor<74x30x80x71xi32>, tensor<74x30x80x71xi32>) -> tensor<74x30x80x71xi32>
    %3 = tosa.reduce_sum %2 {axis = 1 : i32} : (tensor<74x30x80x71xi32>) -> tensor<74x1x80x71xi32>
    %4 = tosa.clamp %3 {min_val = -23 : i32, max_val = 104 : i32} : (tensor<74x1x80x71xi32>) -> tensor<74x1x80x71xi32>
    %5 = tosa.greater %4, %3 : (tensor<74x1x80x71xi32>, tensor<74x1x80x71xi32>) -> tensor<74x1x80x71xi1>
    %6 = tosa.logical_and %5, %5 : (tensor<74x1x80x71xi1>, tensor<74x1x80x71xi1>) -> tensor<74x1x80x71xi1>
    %7 = tosa.exp %arg1 : (tensor<19x11x82xf32>) -> tensor<19x11x82xf32>
    %8 = tosa.minimum %7, %7 : (tensor<19x11x82xf32>, tensor<19x11x82xf32>) -> tensor<19x11x82xf32>
    %in_zp_9 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_9 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %9 = tosa.negate %8, %in_zp_9, %out_zp_9 : (tensor<19x11x82xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<19x11x82xf32>
    %10 = tosa.reduce_min %7 {axis = 0 : i32} : (tensor<19x11x82xf32>) -> tensor<1x11x82xf32>
    %11 = tosa.ceil %10 : (tensor<1x11x82xf32>) -> tensor<1x11x82xf32>
    return %6, %9, %11 : tensor<74x1x80x71xi1>, tensor<19x11x82xf32>, tensor<1x11x82xf32>
  }
}
