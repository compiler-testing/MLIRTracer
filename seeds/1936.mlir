module {
  func.func @main(%arg0: tensor<85x37xf32>, %arg1: tensor<2x22xi8>, %arg2: tensor<64x41x97x33x92xi1>, %arg3: tensor<64x41x1x1x92xi1>) -> (tensor<1x37xf32>, tensor<85x37xf32>, tensor<2x1xi8>, tensor<64x41x97x33x92xi1>, tensor<4x10xi8>) {
    %0 = tosa.rsqrt %arg0 : (tensor<85x37xf32>) -> tensor<85x37xf32>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<85x37xf32>) -> tensor<1x37xf32>
    %2 = tosa.clz %arg1 : (tensor<2x22xi8>) -> tensor<2x22xi8>
    %3 = tosa.bitwise_xor %2, %2 : (tensor<2x22xi8>, tensor<2x22xi8>) -> tensor<2x22xi8>
    %4 = tosa.ceil %0 : (tensor<85x37xf32>) -> tensor<85x37xf32>
    %s_5_start = tosa.const_shape {values = dense<[ 0, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_5_size = tosa.const_shape {values = dense<[ 4, 10 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.slice %2, %s_5_start, %s_5_size : (tensor<2x22xi8>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<4x10xi8>
    %6 = tosa.reduce_max %3 {axis = 1 : i32} : (tensor<2x22xi8>) -> tensor<2x1xi8>
    %7 = tosa.logical_or %arg2, %arg3 : (tensor<64x41x97x33x92xi1>, tensor<64x41x1x1x92xi1>) -> tensor<64x41x97x33x92xi1>
    %8 = tosa.sub %5, %5 : (tensor<4x10xi8>, tensor<4x10xi8>) -> tensor<4x10xi8>
    return %1, %4, %6, %7, %8 : tensor<1x37xf32>, tensor<85x37xf32>, tensor<2x1xi8>, tensor<64x41x97x33x92xi1>, tensor<4x10xi8>
  }
}
