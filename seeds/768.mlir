module {
  func.func @main(%arg0: tensor<59x55x37xf32>) -> tensor<1x1xi32> {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<59x55x37xf32>) -> tensor<1x55x37xf32>
    %1 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<1x55x37xf32>) -> tensor<1x1x37xf32>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<1x1x37xf32>) -> tensor<1x1x37xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<1x1x37xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x37xf32>
    %4 = tosa.argmax %3 {axis = 2 : i32} : (tensor<1x1x37xf32>) -> tensor<1x1xi32>
    return %4 : tensor<1x1xi32>
  }
}
