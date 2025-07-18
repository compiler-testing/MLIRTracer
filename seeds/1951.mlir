module {
  func.func @main(%arg0: tensor<10x52xf32>, %arg1: tensor<40xi8>, %arg2: tensor<40xi8>, %arg3: tensor<30x39x6x51x7xi1>, %arg4: tensor<1x1x1x51x7xi1>) -> (tensor<20x52xf32>, tensor<9xi8>, tensor<30x39x6x51x7xi1>) {
    %0 = tosa.tanh %arg0 : (tensor<10x52xf32>) -> tensor<10x52xf32>
    %1 = tosa.identity %0 : (tensor<10x52xf32>) -> tensor<10x52xf32>
    %2 = tosa.logical_left_shift %arg1, %arg2 : (tensor<40xi8>, tensor<40xi8>) -> tensor<40xi8>
    %t_3 = tosa.const_shape {values = dense<[ 2, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.tile %1, %t_3 : (tensor<10x52xf32>, !tosa.shape<2>) -> tensor<20x52xf32>
    %s_4_start = tosa.const_shape {values = dense<[ 31 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_4_size = tosa.const_shape {values = dense<[ 9 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.slice %2, %s_4_start, %s_4_size : (tensor<40xi8>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<9xi8>
    %5 = tosa.logical_and %arg3, %arg4 : (tensor<30x39x6x51x7xi1>, tensor<1x1x1x51x7xi1>) -> tensor<30x39x6x51x7xi1>
    return %3, %4, %5 : tensor<20x52xf32>, tensor<9xi8>, tensor<30x39x6x51x7xi1>
  }
}
