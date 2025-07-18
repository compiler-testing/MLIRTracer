module {
  func.func @main(%arg0: tensor<31x49x32x18xf32>, %arg1: tensor<22x100x70x3xi8>, %arg2: tensor<1x1x70x3xi8>) -> (tensor<93x1x64x1xf32>, tensor<22x100x70x3xi8>) {
    %0 = tosa.exp %arg0 : (tensor<31x49x32x18xf32>) -> tensor<31x49x32x18xf32>
    %t_1 = tosa.const_shape {values = dense<[ 3, 3, 2, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.tile %0, %t_1 : (tensor<31x49x32x18xf32>, !tosa.shape<4>) -> tensor<93x147x64x54xf32>
    %2 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<93x147x64x54xf32>) -> tensor<93x1x64x54xf32>
    %3 = tosa.reduce_max %2 {axis = 3 : i32} : (tensor<93x1x64x54xf32>) -> tensor<93x1x64x1xf32>
    %4 = tosa.logical_left_shift %arg1, %arg2 : (tensor<22x100x70x3xi8>, tensor<1x1x70x3xi8>) -> tensor<22x100x70x3xi8>
    %5 = tosa.bitwise_or %4, %4 : (tensor<22x100x70x3xi8>, tensor<22x100x70x3xi8>) -> tensor<22x100x70x3xi8>
    return %3, %5 : tensor<93x1x64x1xf32>, tensor<22x100x70x3xi8>
  }
}
