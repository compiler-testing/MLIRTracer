module {
  func.func @main(%arg0: tensor<13x90x61xi32>) -> tensor<13x90x1xi32> {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<13x90x61xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<13x90x61xi32>
    %1 = tosa.concat %0, %0 {axis = 2 : i32} : (tensor<13x90x61xi32>, tensor<13x90x61xi32>) -> tensor<13x90x122xi32>
    %2 = tosa.clamp %1 {min_val = -58 : i32, max_val = 57 : i32} : (tensor<13x90x122xi32>) -> tensor<13x90x122xi32>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<13x90x122xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<13x90x122xi32>
    %4 = tosa.reduce_max %3 {axis = 2 : i32} : (tensor<13x90x122xi32>) -> tensor<13x90x1xi32>
    return %4 : tensor<13x90x1xi32>
  }
}
