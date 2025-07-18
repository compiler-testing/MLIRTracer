module {
  func.func @main(%arg0: tensor<54x25x41x35x25x65xf32>, %arg1: tensor<87x3x45xi8>) -> (tensor<54x25x41x35x25x65xf32>, tensor<1x6x45xi8>, tensor<54x25x41x35x25x65xf32>, tensor<11x11x2xi8>) {
    %0 = tosa.tanh %arg0 : (tensor<54x25x41x35x25x65xf32>) -> tensor<54x25x41x35x25x65xf32>
    %t_1 = tosa.const_shape {values = dense<[ 3, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %arg1, %t_1 : (tensor<87x3x45xi8>, !tosa.shape<3>) -> tensor<261x6x45xi8>
    %2 = tosa.minimum %1, %1 : (tensor<261x6x45xi8>, tensor<261x6x45xi8>) -> tensor<261x6x45xi8>
    %3 = tosa.ceil %0 : (tensor<54x25x41x35x25x65xf32>) -> tensor<54x25x41x35x25x65xf32>
    %4 = tosa.maximum %3, %3 : (tensor<54x25x41x35x25x65xf32>, tensor<54x25x41x35x25x65xf32>) -> tensor<54x25x41x35x25x65xf32>
    %5 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<261x6x45xi8>) -> tensor<1x6x45xi8>
    %6 = tosa.floor %3 : (tensor<54x25x41x35x25x65xf32>) -> tensor<54x25x41x35x25x65xf32>
    %s_7_start = tosa.const_shape {values = dense<[ 83, 0, 43 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_7_size = tosa.const_shape {values = dense<[ 11, 11, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %7 = tosa.slice %2, %s_7_start, %s_7_size : (tensor<261x6x45xi8>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<11x11x2xi8>
    return %4, %5, %6, %7 : tensor<54x25x41x35x25x65xf32>, tensor<1x6x45xi8>, tensor<54x25x41x35x25x65xf32>, tensor<11x11x2xi8>
  }
}
