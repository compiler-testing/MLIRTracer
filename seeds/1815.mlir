module {
  func.func @main(%arg0: tensor<71x70x83x77x39xi1>, %arg1: tensor<71x70x1x1x39xi1>, %arg2: tensor<67x48xi64>, %arg3: tensor<67x48xi64>, %arg4: tensor<88xf32>) -> (tensor<71x70x83x77x39xi1>, tensor<88xf32>, tensor<1x9xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<71x70x83x77x39xi1>, tensor<71x70x1x1x39xi1>) -> tensor<71x70x83x77x39xi1>
    %1 = tosa.greater_equal %arg2, %arg3 : (tensor<67x48xi64>, tensor<67x48xi64>) -> tensor<67x48xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 43, 21 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_2_size = tosa.const_shape {values = dense<[ 11, 9 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<67x48xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<11x9xi1>
    %3 = tosa.bitwise_and %0, %0 : (tensor<71x70x83x77x39xi1>, tensor<71x70x83x77x39xi1>) -> tensor<71x70x83x77x39xi1>
    %4 = tosa.tanh %arg4 : (tensor<88xf32>) -> tensor<88xf32>
    %5 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<11x9xi1>) -> tensor<1x9xi1>
    return %3, %4, %5 : tensor<71x70x83x77x39xi1>, tensor<88xf32>, tensor<1x9xi1>
  }
}
