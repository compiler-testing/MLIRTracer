module {
  func.func @main(%arg0: tensor<7x74x14x53x19x17xi64>, %arg1: tensor<1x1x14x53x1x17xi64>, %arg2: tensor<18x91x69x63x81x67xf32>, %arg3: tensor<1x1x1x1x81x1xf32>, %arg4: tensor<24x18xf32>, %arg5: tensor<24x18xf32>) -> (tensor<7x74x14x53x19x17xi1>, tensor<18x91x69x63x81x67xi1>, tensor<1x18xi1>, tensor<18x91x69x63x81x67xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<7x74x14x53x19x17xi64>, tensor<1x1x14x53x1x17xi64>) -> tensor<7x74x14x53x19x17xi1>
    %1 = tosa.greater_equal %arg2, %arg3 : (tensor<18x91x69x63x81x67xf32>, tensor<1x1x1x1x81x1xf32>) -> tensor<18x91x69x63x81x67xi1>
    %2 = tosa.greater_equal %arg4, %arg5 : (tensor<24x18xf32>, tensor<24x18xf32>) -> tensor<24x18xi1>
    %3 = tosa.reverse %2 {axis = 1 : i32} : (tensor<24x18xi1>) -> tensor<24x18xi1>
    %4 = tosa.bitwise_and %1, %1 : (tensor<18x91x69x63x81x67xi1>, tensor<18x91x69x63x81x67xi1>) -> tensor<18x91x69x63x81x67xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %1, %in_zp_5, %out_zp_5 : (tensor<18x91x69x63x81x67xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<18x91x69x63x81x67xi1>
    %6 = tosa.reduce_max %3 {axis = 0 : i32} : (tensor<24x18xi1>) -> tensor<1x18xi1>
    %7 = tosa.logical_xor %6, %6 : (tensor<1x18xi1>, tensor<1x18xi1>) -> tensor<1x18xi1>
    %8 = tosa.sub %5, %1 : (tensor<18x91x69x63x81x67xi1>, tensor<18x91x69x63x81x67xi1>) -> tensor<18x91x69x63x81x67xi1>
    return %0, %4, %7, %8 : tensor<7x74x14x53x19x17xi1>, tensor<18x91x69x63x81x67xi1>, tensor<1x18xi1>, tensor<18x91x69x63x81x67xi1>
  }
}
