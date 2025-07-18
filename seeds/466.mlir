module {
  func.func @main(%arg0: tensor<70x36x14xi1>, %arg1: tensor<1x36x14xi1>, %arg2: tensor<f32>) -> (tensor<70x36x1xi1>, tensor<f32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<70x36x14xi1>, tensor<1x36x14xi1>) -> tensor<70x36x14xi1>
    %1 = tosa.reduce_product %0 {axis = 2 : i32} : (tensor<70x36x14xi1>) -> tensor<70x36x1xi1>
    %2 = tosa.sub %1, %1 : (tensor<70x36x1xi1>, tensor<70x36x1xi1>) -> tensor<70x36x1xi1>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<70x36x1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<70x36x1xi1>
    %4 = tosa.log %arg2 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.rsqrt %4 : (tensor<f32>) -> tensor<f32>
    return %3, %5 : tensor<70x36x1xi1>, tensor<f32>
  }
}
