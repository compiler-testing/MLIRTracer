module {
  func.func @main(%arg0: tensor<22xi8>, %arg1: tensor<22xi8>, %arg2: tensor<6x1x13x35x25x55xf32>, %arg3: tensor<3x32x73x48xi1>) -> (tensor<3x1x73x48xi1>, tensor<22xi8>, tensor<5x3x12x10xi1>, tensor<6x1x13x35x25x55xf32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<22xi8>, tensor<22xi8>) -> tensor<22xi8>
    %1 = tosa.exp %arg2 : (tensor<6x1x13x35x25x55xf32>) -> tensor<6x1x13x35x25x55xf32>
    %2 = tosa.reduce_any %arg3 {axis = 1 : i32} : (tensor<3x32x73x48xi1>) -> tensor<3x1x73x48xi1>
    %3 = tosa.clz %2 : (tensor<3x1x73x48xi1>) -> tensor<3x1x73x48xi1>
    %4 = tosa.reverse %0 {axis = 0 : i32} : (tensor<22xi8>) -> tensor<22xi8>
    %s_5_start = tosa.const_shape {values = dense<[ 0, 0, 3, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_5_size = tosa.const_shape {values = dense<[ 5, 3, 12, 10 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.slice %2, %s_5_start, %s_5_size : (tensor<3x1x73x48xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x3x12x10xi1>
    %6 = tosa.reciprocal %1 : (tensor<6x1x13x35x25x55xf32>) -> tensor<6x1x13x35x25x55xf32>
    return %3, %4, %5, %6 : tensor<3x1x73x48xi1>, tensor<22xi8>, tensor<5x3x12x10xi1>, tensor<6x1x13x35x25x55xf32>
  }
}
