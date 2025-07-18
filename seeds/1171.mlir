module {
  func.func @main(%arg0: tensor<77x39xi1>, %arg1: tensor<31xf32>, %arg2: tensor<31xf32>) -> (tensor<31xi1>, tensor<9x6xi1>) {
    %0 = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<77x39xi1>) -> tensor<77x1xi1>
    %1 = tosa.greater %arg1, %arg2 : (tensor<31xf32>, tensor<31xf32>) -> tensor<31xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 17, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_2_size = tosa.const_shape {values = dense<[ 9, 6 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.slice %0, %s_2_start, %s_2_size : (tensor<77x1xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<9x6xi1>
    return %1, %2 : tensor<31xi1>, tensor<9x6xi1>
  }
}
