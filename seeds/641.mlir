module {
  func.func @main(%arg0: tensor<72x10x3x14xi1>, %arg1: tensor<1x10x1x1xi1>, %arg2: tensor<55x85x81x11x47x54xi64>, %arg3: tensor<55x85x81x11x47x1xi64>) -> (tensor<72x10x3x1xi1>, tensor<55x85x81x11x47x54xi64>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<72x10x3x14xi1>, tensor<1x10x1x1xi1>) -> tensor<72x10x3x14xi1>
    %1 = tosa.reduce_sum %0 {axis = 3 : i32} : (tensor<72x10x3x14xi1>) -> tensor<72x10x3x1xi1>
    %2 = tosa.minimum %arg2, %arg3 : (tensor<55x85x81x11x47x54xi64>, tensor<55x85x81x11x47x1xi64>) -> tensor<55x85x81x11x47x54xi64>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<72x10x3x1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<72x10x3x1xi1>
    %4 = tosa.bitwise_and %2, %2 : (tensor<55x85x81x11x47x54xi64>, tensor<55x85x81x11x47x54xi64>) -> tensor<55x85x81x11x47x54xi64>
    return %3, %4 : tensor<72x10x3x1xi1>, tensor<55x85x81x11x47x54xi64>
  }
}
