module {
  func.func @main(%arg0: tensor<24x71x88x74xi64>, %arg1: tensor<18x1x8x66xi1>) -> (tensor<1x71x88x74xi64>, tensor<18x1x8x66xi1>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<24x71x88x74xi64>) -> tensor<1x71x88x74xi64>
    %1 = tosa.logical_not %arg1 : (tensor<18x1x8x66xi1>) -> tensor<18x1x8x66xi1>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<18x1x8x66xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<18x1x8x66xi1>
    return %0, %2 : tensor<1x71x88x74xi64>, tensor<18x1x8x66xi1>
  }
}
