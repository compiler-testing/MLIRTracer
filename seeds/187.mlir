module {
  func.func @main(%arg0: tensor<41x77xi64>, %arg1: tensor<89x32xf32>, %arg2: tensor<1x1xf32>) -> (tensor<89x32xi1>, tensor<89x32xf32>, tensor<7x1x1x11xi64>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<41x77xi64>) -> tensor<1x77xi64>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<1x77xi64>, tensor<1x77xi64>) -> tensor<1x77xi64>
    %2 = tosa.pow %arg1, %arg2 : (tensor<89x32xf32>, tensor<1x1xf32>) -> tensor<89x32xf32>
    %r_3 = tosa.const_shape {values = dense<[ 7, 1, 1, 11 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.reshape %1, %r_3 : (tensor<1x77xi64>, !tosa.shape<4>) -> tensor<7x1x1x11xi64>
    %4 = tosa.equal %2, %2 : (tensor<89x32xf32>, tensor<89x32xf32>) -> tensor<89x32xi1>
    %5 = tosa.bitwise_and %3, %3 : (tensor<7x1x1x11xi64>, tensor<7x1x1x11xi64>) -> tensor<7x1x1x11xi64>
    %in_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %6 = tosa.negate %5, %in_zp_6, %out_zp_6 : (tensor<7x1x1x11xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<7x1x1x11xi64>
    %7 = tosa.reciprocal %2 : (tensor<89x32xf32>) -> tensor<89x32xf32>
    %8 = tosa.abs %6 : (tensor<7x1x1x11xi64>) -> tensor<7x1x1x11xi64>
    return %4, %7, %8 : tensor<89x32xi1>, tensor<89x32xf32>, tensor<7x1x1x11xi64>
  }
}
