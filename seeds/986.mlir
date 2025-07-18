module {
  func.func @main(%arg0: tensor<39x5xi64>, %arg1: tensor<f32>) -> (tensor<39x5xi64>, tensor<f32>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<39x5xi64>) -> tensor<39x5xi64>
    %1 = tosa.bitwise_not %0 : (tensor<39x5xi64>) -> tensor<39x5xi64>
    %2 = tosa.floor %arg1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.bitwise_or %1, %1 : (tensor<39x5xi64>, tensor<39x5xi64>) -> tensor<39x5xi64>
    %in_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.negate %2, %in_zp_4, %out_zp_4 : (tensor<f32>, tensor<1xf32>, tensor<1xf32>) -> tensor<f32>
    return %3, %4 : tensor<39x5xi64>, tensor<f32>
  }
}
