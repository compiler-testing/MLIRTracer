module {
  func.func @main(%arg0: tensor<53xi64>, %arg1: tensor<53xi64>, %arg2: tensor<24x51xf32>) -> (tensor<1xi64>, tensor<24x51xf32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<53xi64>, tensor<53xi64>) -> tensor<53xi64>
    %1 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0>} : (tensor<53xi64>) -> tensor<53xi64>
    %3 = tosa.ceil %arg2 : (tensor<24x51xf32>) -> tensor<24x51xf32>
    %4 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<53xi64>) -> tensor<1xi64>
    %in_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5 = tosa.negate %3, %in_zp_5, %out_zp_5 : (tensor<24x51xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<24x51xf32>
    return %4, %5 : tensor<1xi64>, tensor<24x51xf32>
  }
}
