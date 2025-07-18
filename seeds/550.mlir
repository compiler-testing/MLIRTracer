module {
  func.func @main(%arg0: tensor<7x92x76xf32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> (tensor<7x92x76xf32>, tensor<i64>) {
    %0 = tosa.ceil %arg0 : (tensor<7x92x76xf32>) -> tensor<7x92x76xf32>
    %1 = tosa.bitwise_and %arg1, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<i64>, tensor<1xi64>, tensor<1xi64>) -> tensor<i64>
    return %0, %2 : tensor<7x92x76xf32>, tensor<i64>
  }
}
