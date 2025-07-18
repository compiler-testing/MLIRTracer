module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<66xf32>) -> (tensor<i64>, tensor<1xf32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<i64>, tensor<1xi64>, tensor<1xi64>) -> tensor<i64>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = tosa.sigmoid %arg1 : (tensor<66xf32>) -> tensor<66xf32>
    %3 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<66xf32>) -> tensor<1xf32>
    return %1, %3 : tensor<i64>, tensor<1xf32>
  }
}
