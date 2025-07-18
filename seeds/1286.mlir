module {
  func.func @main(%arg0: tensor<17x27x49xi64>) -> tensor<17x1x49xi64> {
    %0 = tosa.bitwise_not %arg0 : (tensor<17x27x49xi64>) -> tensor<17x27x49xi64>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<17x27x49xi64>, tensor<17x27x49xi64>) -> tensor<17x27x49xi64>
    %2 = tosa.clamp %1 {min_val = -62 : i64, max_val = -28 : i64} : (tensor<17x27x49xi64>) -> tensor<17x27x49xi64>
    %3 = tosa.sub %2, %2 : (tensor<17x27x49xi64>, tensor<17x27x49xi64>) -> tensor<17x27x49xi64>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4 = tosa.negate %3, %in_zp_4, %out_zp_4 : (tensor<17x27x49xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<17x27x49xi64>
    %5 = tosa.clz %4 : (tensor<17x27x49xi64>) -> tensor<17x27x49xi64>
    %6 = tosa.reduce_min %5 {axis = 1 : i32} : (tensor<17x27x49xi64>) -> tensor<17x1x49xi64>
    return %6 : tensor<17x1x49xi64>
  }
}
