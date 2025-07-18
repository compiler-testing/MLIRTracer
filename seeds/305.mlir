module {
  func.func @main(%arg0: tensor<45xf32>, %arg1: tensor<6x62xi1>, %arg2: tensor<60x3x94x17x9xi32>, %arg3: tensor<60x3x1x1x1xi32>) -> (tensor<45xf32>, tensor<1x1xi1>, tensor<45xf32>, tensor<12x3xi1>, tensor<45xf32>, tensor<60x3x94x17x9xi32>) {
    %0 = tosa.tanh %arg0 : (tensor<45xf32>) -> tensor<45xf32>
    %in_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<45xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<45xf32>
    %2 = tosa.tanh %1 : (tensor<45xf32>) -> tensor<45xf32>
    %3 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<6x62xi1>) -> tensor<1x62xi1>
    %4 = tosa.floor %2 : (tensor<45xf32>) -> tensor<45xf32>
    %5 = tosa.reduce_min %3 {axis = 1 : i32} : (tensor<1x62xi1>) -> tensor<1x1xi1>
    %6 = tosa.arithmetic_right_shift %5, %5 {round = false} : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %7 = tosa.logical_left_shift %5, %6 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %8 = tosa.reduce_max %6 {axis = 0 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %s_9_start = tosa.const_shape {values = dense<[ 0, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_9_size = tosa.const_shape {values = dense<[ 12, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %9 = tosa.slice %7, %s_9_start, %s_9_size : (tensor<1x1xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<12x3xi1>
    %10 = tosa.pow %2, %0 : (tensor<45xf32>, tensor<45xf32>) -> tensor<45xf32>
    %11 = tosa.arithmetic_right_shift %9, %9 {round = false} : (tensor<12x3xi1>, tensor<12x3xi1>) -> tensor<12x3xi1>
    %12 = tosa.exp %0 : (tensor<45xf32>) -> tensor<45xf32>
    %13 = tosa.intdiv %arg2, %arg3 : (tensor<60x3x94x17x9xi32>, tensor<60x3x1x1x1xi32>) -> tensor<60x3x94x17x9xi32>
    return %4, %8, %10, %11, %12, %13 : tensor<45xf32>, tensor<1x1xi1>, tensor<45xf32>, tensor<12x3xi1>, tensor<45xf32>, tensor<60x3x94x17x9xi32>
  }
}
