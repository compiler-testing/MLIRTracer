module {
  func.func @main(%arg0: tensor<94x54x95x26x80x27xi1>, %arg1: tensor<1x1x95x26x80x1xi1>, %arg2: tensor<65x31xi32>, %arg3: tensor<65x1xi32>, %arg4: tensor<91x85x50x29x93xf32>) -> (tensor<65x31xi32>, tensor<94x54x95x26x80x27xi1>, tensor<1x31xi32>, tensor<91x85x50x29x93xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<94x54x95x26x80x27xi1>, tensor<1x1x95x26x80x1xi1>) -> tensor<94x54x95x26x80x27xi1>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<65x31xi32>, tensor<65x1xi32>) -> tensor<65x31xi32>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<65x31xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<65x31xi32>
    %3 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<65x31xi32>) -> tensor<1x31xi32>
    %4 = tosa.logical_left_shift %2, %1 : (tensor<65x31xi32>, tensor<65x31xi32>) -> tensor<65x31xi32>
    %5 = tosa.logical_or %0, %0 : (tensor<94x54x95x26x80x27xi1>, tensor<94x54x95x26x80x27xi1>) -> tensor<94x54x95x26x80x27xi1>
    %6 = tosa.logical_right_shift %3, %3 : (tensor<1x31xi32>, tensor<1x31xi32>) -> tensor<1x31xi32>
    %7 = tosa.tanh %arg4 : (tensor<91x85x50x29x93xf32>) -> tensor<91x85x50x29x93xf32>
    %8 = tosa.log %7 : (tensor<91x85x50x29x93xf32>) -> tensor<91x85x50x29x93xf32>
    return %4, %5, %6, %8 : tensor<65x31xi32>, tensor<94x54x95x26x80x27xi1>, tensor<1x31xi32>, tensor<91x85x50x29x93xf32>
  }
}
