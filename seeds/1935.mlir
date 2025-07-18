module {
  func.func @main(%arg0: tensor<63x94x78x81x54xi16>, %arg1: tensor<1x94x78x1x1xi16>, %arg2: tensor<98x94xf32>, %arg3: tensor<92xi1>, %arg4: tensor<92xi1>) -> (tensor<63x94x78x81x54xi16>, tensor<98xi32>, tensor<92xi1>, tensor<8x8xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<63x94x78x81x54xi16>, tensor<1x94x78x1x1xi16>) -> tensor<63x94x78x81x54xi16>
    %1 = tosa.identity %0 : (tensor<63x94x78x81x54xi16>) -> tensor<63x94x78x81x54xi16>
    %2 = tosa.reciprocal %arg2 : (tensor<98x94xf32>) -> tensor<98x94xf32>
    %3 = tosa.tanh %2 : (tensor<98x94xf32>) -> tensor<98x94xf32>
    %4 = tosa.rsqrt %3 : (tensor<98x94xf32>) -> tensor<98x94xf32>
    %5 = tosa.argmax %4 {axis = 1 : i32} : (tensor<98x94xf32>) -> tensor<98xi32>
    %6 = tosa.rsqrt %2 : (tensor<98x94xf32>) -> tensor<98x94xf32>
    %7 = tosa.logical_or %arg3, %arg4 : (tensor<92xi1>, tensor<92xi1>) -> tensor<92xi1>
    %s_8_start = tosa.const_shape {values = dense<[ 83, 79 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_8_size = tosa.const_shape {values = dense<[ 8, 8 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %8 = tosa.slice %6, %s_8_start, %s_8_size : (tensor<98x94xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<8x8xf32>
    return %1, %5, %7, %8 : tensor<63x94x78x81x54xi16>, tensor<98xi32>, tensor<92xi1>, tensor<8x8xf32>
  }
}
