module {
  func.func @main(%arg0: tensor<25x60x60x53xf32>, %arg1: tensor<50x58xi1>, %arg2: tensor<1x58xi1>, %arg3: tensor<26x79x66x3x94x17xi32>, %arg4: tensor<26x1x1x3x94x1xi32>) -> (tensor<50x58xi1>, tensor<8x3x9x9xf32>, tensor<26x79x66x3x94x17xi32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<25x60x60x53xf32>) -> tensor<25x60x60x53xf32>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<50x58xi1>, tensor<1x58xi1>) -> tensor<50x58xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 4, 7, 7, 15 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_2_size = tosa.const_shape {values = dense<[ 8, 3, 9, 9 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.slice %0, %s_2_start, %s_2_size : (tensor<25x60x60x53xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<8x3x9x9xf32>
    %3 = tosa.sigmoid %2 : (tensor<8x3x9x9xf32>) -> tensor<8x3x9x9xf32>
    %4 = tosa.sigmoid %3 : (tensor<8x3x9x9xf32>) -> tensor<8x3x9x9xf32>
    %5 = tosa.intdiv %arg3, %arg4 : (tensor<26x79x66x3x94x17xi32>, tensor<26x1x1x3x94x1xi32>) -> tensor<26x79x66x3x94x17xi32>
    return %1, %4, %5 : tensor<50x58xi1>, tensor<8x3x9x9xf32>, tensor<26x79x66x3x94x17xi32>
  }
}
