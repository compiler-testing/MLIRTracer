module {
  func.func @main(%arg0: tensor<100x61x33xi32>, %arg1: tensor<97x75x64xi1>, %arg2: tensor<1x75x64xi1>) -> (tensor<9x10x8xi1>, tensor<100x33xi32>) {
    %0 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<100x61x33xi32>) -> tensor<100x33xi32>
    %1 = tosa.intdiv %0, %0 : (tensor<100x33xi32>, tensor<100x33xi32>) -> tensor<100x33xi32>
    %2 = tosa.logical_left_shift %1, %0 : (tensor<100x33xi32>, tensor<100x33xi32>) -> tensor<100x33xi32>
    %3 = tosa.intdiv %2, %2 : (tensor<100x33xi32>, tensor<100x33xi32>) -> tensor<100x33xi32>
    %4 = tosa.logical_and %arg1, %arg2 : (tensor<97x75x64xi1>, tensor<1x75x64xi1>) -> tensor<97x75x64xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 12, 39, 56 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_5_size = tosa.const_shape {values = dense<[ 9, 10, 8 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<97x75x64xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<9x10x8xi1>
    %6 = tosa.reverse %3 {axis = 0 : i32} : (tensor<100x33xi32>) -> tensor<100x33xi32>
    return %5, %6 : tensor<9x10x8xi1>, tensor<100x33xi32>
  }
}
