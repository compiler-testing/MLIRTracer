module {
  func.func @main(%arg0: tensor<95xf32>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<96x19x82x19xi1>) -> (tensor<i64>, tensor<95xf32>, tensor<96x1x82x1xi1>, tensor<12x8x5x7xi1>, tensor<96x1x82x1xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<95xf32>) -> tensor<95xf32>
    %1 = tosa.bitwise_and %arg1, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = tosa.clamp %1 {min_val = 14 : i64, max_val = 32 : i64} : (tensor<i64>) -> tensor<i64>
    %3 = tosa.reduce_any %arg3 {axis = 3 : i32} : (tensor<96x19x82x19xi1>) -> tensor<96x19x82x1xi1>
    %4 = tosa.reduce_min %3 {axis = 1 : i32} : (tensor<96x19x82x1xi1>) -> tensor<96x1x82x1xi1>
    %5 = tosa.exp %0 : (tensor<95xf32>) -> tensor<95xf32>
    %in_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %6 = tosa.negate %4, %in_zp_6, %out_zp_6 : (tensor<96x1x82x1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<96x1x82x1xi1>
    %7 = tosa.arithmetic_right_shift %4, %4 {round = false} : (tensor<96x1x82x1xi1>, tensor<96x1x82x1xi1>) -> tensor<96x1x82x1xi1>
    %8 = tosa.clz %6 : (tensor<96x1x82x1xi1>) -> tensor<96x1x82x1xi1>
    %9 = tosa.logical_and %7, %4 : (tensor<96x1x82x1xi1>, tensor<96x1x82x1xi1>) -> tensor<96x1x82x1xi1>
    %s_10_start = tosa.const_shape {values = dense<[ 32, 0, 33, 0 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_10_size = tosa.const_shape {values = dense<[ 12, 8, 5, 7 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.slice %8, %s_10_start, %s_10_size : (tensor<96x1x82x1xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<12x8x5x7xi1>
    %11 = tosa.bitwise_and %7, %7 : (tensor<96x1x82x1xi1>, tensor<96x1x82x1xi1>) -> tensor<96x1x82x1xi1>
    %12 = tosa.bitwise_not %11 : (tensor<96x1x82x1xi1>) -> tensor<96x1x82x1xi1>
    return %2, %5, %9, %10, %12 : tensor<i64>, tensor<95xf32>, tensor<96x1x82x1xi1>, tensor<12x8x5x7xi1>, tensor<96x1x82x1xi1>
  }
}
