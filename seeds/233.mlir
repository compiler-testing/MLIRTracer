module {
  func.func @main(%arg0: tensor<93x44xi32>, %arg1: tensor<6x93x92xf32>) -> (tensor<7x11xi32>, tensor<6x93x92xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 1, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.tile %arg0, %t_0 : (tensor<93x44xi32>, !tosa.shape<2>) -> tensor<93x132xi32>
    %s_1_start = tosa.const_shape {values = dense<[ 74, 41 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_1_size = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<93x132xi32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x2xi32>
    %2 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<1x2xi32>) -> tensor<1x1xi32>
    %3 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<1x1xi32>) -> tensor<1x1xi32>
    %s_4_start = tosa.const_shape {values = dense<[ 0, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_4_size = tosa.const_shape {values = dense<[ 7, 11 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.slice %3, %s_4_start, %s_4_size : (tensor<1x1xi32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<7x11xi32>
    %5 = tosa.bitwise_or %4, %4 : (tensor<7x11xi32>, tensor<7x11xi32>) -> tensor<7x11xi32>
    %6 = tosa.reciprocal %arg1 : (tensor<6x93x92xf32>) -> tensor<6x93x92xf32>
    return %5, %6 : tensor<7x11xi32>, tensor<6x93x92xf32>
  }
}
