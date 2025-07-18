module {
  func.func @main(%arg0: tensor<52x92x43xf32>, %arg1: tensor<52x1x43xf32>, %arg2: tensor<54x51x10xf32>) -> (tensor<54x51x10xf32>, tensor<52x92x43xi1>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<52x92x43xf32>, tensor<52x1x43xf32>) -> tensor<52x92x43xi1>
    %1 = tosa.clz %0 : (tensor<52x92x43xi1>) -> tensor<52x92x43xi1>
    %2 = tosa.exp %arg2 : (tensor<54x51x10xf32>) -> tensor<54x51x10xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<52x92x43xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<52x92x43xi1>
    return %2, %3 : tensor<54x51x10xf32>, tensor<52x92x43xi1>
  }
}
