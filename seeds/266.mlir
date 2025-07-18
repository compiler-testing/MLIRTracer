module {
  func.func @main(%arg0: tensor<29x94x57x34x82x72xf32>, %arg1: tensor<i1>, %arg2: tensor<i1>, %arg3: tensor<55x27x24xi64>) -> (tensor<i1>, tensor<29x94x57x34x82x72xi1>, tensor<1x1x24xi64>) {
    %0 = tosa.rsqrt %arg0 : (tensor<29x94x57x34x82x72xf32>) -> tensor<29x94x57x34x82x72xf32>
    %1 = tosa.logical_right_shift %arg1, %arg2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.pow %0, %0 : (tensor<29x94x57x34x82x72xf32>, tensor<29x94x57x34x82x72xf32>) -> tensor<29x94x57x34x82x72xf32>
    %3 = tosa.clz %1 : (tensor<i1>) -> tensor<i1>
    %4 = tosa.pow %2, %2 : (tensor<29x94x57x34x82x72xf32>, tensor<29x94x57x34x82x72xf32>) -> tensor<29x94x57x34x82x72xf32>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %3, %in_zp_5, %out_zp_5 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %6 = tosa.bitwise_or %5, %5 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.logical_not %6 : (tensor<i1>) -> tensor<i1>
    %8 = tosa.equal %4, %0 : (tensor<29x94x57x34x82x72xf32>, tensor<29x94x57x34x82x72xf32>) -> tensor<29x94x57x34x82x72xi1>
    %9 = tosa.reduce_max %arg3 {axis = 1 : i32} : (tensor<55x27x24xi64>) -> tensor<55x1x24xi64>
    %10 = tosa.reduce_sum %9 {axis = 0 : i32} : (tensor<55x1x24xi64>) -> tensor<1x1x24xi64>
    return %7, %8, %10 : tensor<i1>, tensor<29x94x57x34x82x72xi1>, tensor<1x1x24xi64>
  }
}
