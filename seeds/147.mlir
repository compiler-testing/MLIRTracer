module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<91xi32>, %arg2: tensor<1xi32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<i64>, tensor<91xi32>, tensor<91xi32>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<i64>) -> tensor<i64>
    %1 = tosa.maximum %arg1, %arg2 : (tensor<91xi32>, tensor<1xi32>) -> tensor<91xi32>
    %2 = tosa.sub %1, %1 : (tensor<91xi32>, tensor<91xi32>) -> tensor<91xi32>
    %3 = tosa.logical_right_shift %0, %0 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %4 = tosa.sub %2, %2 : (tensor<91xi32>, tensor<91xi32>) -> tensor<91xi32>
    %5 = tosa.ceil %arg3 : (tensor<f32>) -> tensor<f32>
    %6 = tosa.bitwise_or %4, %4 : (tensor<91xi32>, tensor<91xi32>) -> tensor<91xi32>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %7 = tosa.negate %3, %in_zp_7, %out_zp_7 : (tensor<i64>, tensor<1xi64>, tensor<1xi64>) -> tensor<i64>
    %t_8 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %8 = tosa.tile %6, %t_8 : (tensor<91xi32>, !tosa.shape<1>) -> tensor<91xi32>
    %9 = tosa.bitwise_and %6, %8 : (tensor<91xi32>, tensor<91xi32>) -> tensor<91xi32>
    %10 = tosa.maximum %8, %4 : (tensor<91xi32>, tensor<91xi32>) -> tensor<91xi32>
    %11 = tosa.abs %9 : (tensor<91xi32>) -> tensor<91xi32>
    %12 = tosa.sub %9, %11 : (tensor<91xi32>, tensor<91xi32>) -> tensor<91xi32>
    return %5, %7, %10, %12 : tensor<f32>, tensor<i64>, tensor<91xi32>, tensor<91xi32>
  }
}
