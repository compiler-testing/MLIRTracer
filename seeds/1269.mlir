module {
  func.func @main(%arg0: tensor<96xi1>, %arg1: tensor<94x27x2x87x72xf32>, %arg2: tensor<43xi32>, %arg3: tensor<1xi32>) -> (tensor<94x27x2x87x72xf32>, tensor<1xi1>, tensor<43xi32>) {
    %0 = tosa.logical_not %arg0 : (tensor<96xi1>) -> tensor<96xi1>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<96xi1>, tensor<96xi1>) -> tensor<96xi1>
    %2 = tosa.log %arg1 : (tensor<94x27x2x87x72xf32>) -> tensor<94x27x2x87x72xf32>
    %3 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<96xi1>) -> tensor<1xi1>
    %4 = tosa.intdiv %arg2, %arg3 : (tensor<43xi32>, tensor<1xi32>) -> tensor<43xi32>
    %5 = tosa.logical_right_shift %4, %4 : (tensor<43xi32>, tensor<43xi32>) -> tensor<43xi32>
    %in_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = tosa.negate %5, %in_zp_6, %out_zp_6 : (tensor<43xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<43xi32>
    return %2, %3, %6 : tensor<94x27x2x87x72xf32>, tensor<1xi1>, tensor<43xi32>
  }
}
