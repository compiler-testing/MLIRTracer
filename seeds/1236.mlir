module {
  func.func @main(%arg0: tensor<84xi1>, %arg1: tensor<7x36x92x55xi32>, %arg2: tensor<1x36x1x1xi32>) -> (tensor<1xi1>, tensor<3x21252x20xi32>) {
    %0 = tosa.logical_not %arg0 : (tensor<84xi1>) -> tensor<84xi1>
    %1 = tosa.logical_or %0, %0 : (tensor<84xi1>, tensor<84xi1>) -> tensor<84xi1>
    %2 = tosa.arithmetic_right_shift %1, %1 {round = false} : (tensor<84xi1>, tensor<84xi1>) -> tensor<84xi1>
    %3 = tosa.reduce_any %2 {axis = 0 : i32} : (tensor<84xi1>) -> tensor<1xi1>
    %t_4 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.tile %3, %t_4 : (tensor<1xi1>, !tosa.shape<1>) -> tensor<1xi1>
    %5 = tosa.logical_xor %4, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.intdiv %arg1, %arg2 : (tensor<7x36x92x55xi32>, tensor<1x36x1x1xi32>) -> tensor<7x36x92x55xi32>
    %7 = tosa.clz %6 : (tensor<7x36x92x55xi32>) -> tensor<7x36x92x55xi32>
    %r_8 = tosa.const_shape {values = dense<[ 3, 21252, 20 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %8 = tosa.reshape %7, %r_8 : (tensor<7x36x92x55xi32>, !tosa.shape<3>) -> tensor<3x21252x20xi32>
    return %5, %8 : tensor<1xi1>, tensor<3x21252x20xi32>
  }
}
