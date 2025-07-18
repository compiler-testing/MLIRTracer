module {
  func.func @main(%arg0: tensor<23x98x29xi1>, %arg1: tensor<22xf32>) -> (tensor<810xi1>, tensor<1xf32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 10, 4, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_0_size = tosa.const_shape {values = dense<[ 10, 2, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<23x98x29xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<10x2x3xi1>
    %1 = tosa.tanh %arg1 : (tensor<22xf32>) -> tensor<22xf32>
    %2 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<10x2x3xi1>) -> tensor<10x1x3xi1>
    %t_3 = tosa.const_shape {values = dense<[ 3, 3, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.tile %2, %t_3 : (tensor<10x1x3xi1>, !tosa.shape<3>) -> tensor<30x3x9xi1>
    %r_4 = tosa.const_shape {values = dense<[ 810 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.reshape %3, %r_4 : (tensor<30x3x9xi1>, !tosa.shape<1>) -> tensor<810xi1>
    %5 = tosa.abs %4 : (tensor<810xi1>) -> tensor<810xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 19 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_6_size = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.slice %1, %s_6_start, %s_6_size : (tensor<22xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf32>
    %7 = tosa.reduce_product %6 {axis = 0 : i32} : (tensor<3xf32>) -> tensor<1xf32>
    %8 = tosa.logical_and %5, %4 : (tensor<810xi1>, tensor<810xi1>) -> tensor<810xi1>
    %9 = tosa.tanh %7 : (tensor<1xf32>) -> tensor<1xf32>
    return %8, %9 : tensor<810xi1>, tensor<1xf32>
  }
}
