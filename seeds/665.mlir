module {
  func.func @main(%arg0: tensor<65x68xf32>, %arg1: tensor<4x22xi64>, %arg2: tensor<1x1xi64>) -> (tensor<65x68xi1>, tensor<1x22xi64>) {
    %0 = tosa.floor %arg0 : (tensor<65x68xf32>) -> tensor<65x68xf32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<4x22xi64>, tensor<1x1xi64>) -> tensor<4x22xi64>
    %in_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.negate %0, %in_zp_2, %out_zp_2 : (tensor<65x68xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<65x68xf32>
    %3 = tosa.greater_equal %2, %2 : (tensor<65x68xf32>, tensor<65x68xf32>) -> tensor<65x68xi1>
    %4 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<4x22xi64>) -> tensor<1x22xi64>
    return %3, %4 : tensor<65x68xi1>, tensor<1x22xi64>
  }
}
