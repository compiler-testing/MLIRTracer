module {
  func.func @main(%arg0: tensor<15x48xf32>, %arg1: tensor<2x2xi64>) -> tensor<4x1xi1> {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<15x48xf32>, !tosa.shape<4>, tensor<1xf32>) -> tensor<15x48xf32>
    %1 = tosa.exp %0 : (tensor<15x48xf32>) -> tensor<15x48xf32>
    %t_2 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.tile %1, %t_2 : (tensor<15x48xf32>, !tosa.shape<2>) -> tensor<45x96xf32>
    %3 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<45x96xf32>) -> tensor<1x96xf32>
    %4 = tosa.abs %3 : (tensor<1x96xf32>) -> tensor<1x96xf32>
    %5 = tosa.equal %4, %4 : (tensor<1x96xf32>, tensor<1x96xf32>) -> tensor<1x96xi1>
    %6 = tosa.logical_xor %5, %5 : (tensor<1x96xi1>, tensor<1x96xi1>) -> tensor<1x96xi1>
    %7 = tosa.logical_xor %6, %5 : (tensor<1x96xi1>, tensor<1x96xi1>) -> tensor<1x96xi1>
    %8 = tosa.logical_right_shift %7, %7 : (tensor<1x96xi1>, tensor<1x96xi1>) -> tensor<1x96xi1>
    %s_9_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_9_size = tosa.const_shape {values = dense<[ 4, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %9 = tosa.slice %8, %s_9_start, %s_9_size : (tensor<1x96xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<4x3xi1>
    %10 = tosa.reduce_sum %9 {axis = 1 : i32} : (tensor<4x3xi1>) -> tensor<4x1xi1>
    %11 = tosa.sub %10, %10 : (tensor<4x1xi1>, tensor<4x1xi1>) -> tensor<4x1xi1>
    %12 = tosa.bitwise_xor %11, %11 : (tensor<4x1xi1>, tensor<4x1xi1>) -> tensor<4x1xi1>
    return %12 : tensor<4x1xi1>
  }
}
