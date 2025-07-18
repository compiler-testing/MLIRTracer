module {
  func.func @main(%arg0: tensor<88x82x46xi8>, %arg1: tensor<81x45xf32>, %arg2: tensor<i1>, %arg3: tensor<i1>, %arg4: tensor<95x39x35x49x48xi32>, %arg5: tensor<1x1x35x1x48xi32>) -> (tensor<162x90xf32>, tensor<1x82x46xi8>, tensor<i1>, tensor<95x39x35x49x48xi32>) {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<88x82x46xi8>) -> tensor<1x82x46xi8>
    %1 = tosa.reciprocal %arg1 : (tensor<81x45xf32>) -> tensor<81x45xf32>
    %t_2 = tosa.const_shape {values = dense<[ 2, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.tile %1, %t_2 : (tensor<81x45xf32>, !tosa.shape<2>) -> tensor<162x90xf32>
    %3 = tosa.logical_left_shift %0, %0 : (tensor<1x82x46xi8>, tensor<1x82x46xi8>) -> tensor<1x82x46xi8>
    %4 = tosa.reduce_max %3 {axis = 0 : i32} : (tensor<1x82x46xi8>) -> tensor<1x82x46xi8>
    %5 = tosa.logical_xor %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %6 = tosa.intdiv %arg4, %arg5 : (tensor<95x39x35x49x48xi32>, tensor<1x1x35x1x48xi32>) -> tensor<95x39x35x49x48xi32>
    return %2, %4, %5, %6 : tensor<162x90xf32>, tensor<1x82x46xi8>, tensor<i1>, tensor<95x39x35x49x48xi32>
  }
}
