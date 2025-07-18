module {
  func.func @main(%arg0: tensor<70x87x5x45xi8>, %arg1: tensor<70x36x5x45xi8>, %arg2: tensor<14x84x45x33x75x8xf32>, %arg3: tensor<1x84x45x33x75x1xf32>, %arg4: tensor<45x25xi1>) -> (tensor<14x84x45x33x75x8xf32>, tensor<70x123x5x45xi8>, tensor<1x25xi1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<70x87x5x45xi8>, tensor<70x36x5x45xi8>) -> tensor<70x123x5x45xi8>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<70x123x5x45xi8>, tensor<70x123x5x45xi8>) -> tensor<70x123x5x45xi8>
    %2 = tosa.clz %1 : (tensor<70x123x5x45xi8>) -> tensor<70x123x5x45xi8>
    %3 = tosa.pow %arg2, %arg3 : (tensor<14x84x45x33x75x8xf32>, tensor<1x84x45x33x75x1xf32>) -> tensor<14x84x45x33x75x8xf32>
    %4 = tosa.logical_right_shift %2, %0 : (tensor<70x123x5x45xi8>, tensor<70x123x5x45xi8>) -> tensor<70x123x5x45xi8>
    %5 = tosa.reduce_any %arg4 {axis = 0 : i32} : (tensor<45x25xi1>) -> tensor<1x25xi1>
    return %3, %4, %5 : tensor<14x84x45x33x75x8xf32>, tensor<70x123x5x45xi8>, tensor<1x25xi1>
  }
}
