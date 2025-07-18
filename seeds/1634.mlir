module {
  func.func @main(%arg0: tensor<86x62x64x69xi64>, %arg1: tensor<86x1x1x1xi64>, %arg2: tensor<62x87xi32>, %arg3: tensor<1x1xi32>) -> (tensor<10x2x4x11xi1>, tensor<62x87xi32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<86x62x64x69xi64>, tensor<86x1x1x1xi64>) -> tensor<86x62x64x69xi1>
    %1 = tosa.reverse %0 {axis = 3 : i32} : (tensor<86x62x64x69xi1>) -> tensor<86x62x64x69xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 14, 28, 18, 27 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_2_size = tosa.const_shape {values = dense<[ 5, 2, 4, 11 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<86x62x64x69xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x2x4x11xi1>
    %3 = tosa.concat %2, %2 {axis = 0 : i32} : (tensor<5x2x4x11xi1>, tensor<5x2x4x11xi1>) -> tensor<10x2x4x11xi1>
    %4 = tosa.logical_xor %3, %3 : (tensor<10x2x4x11xi1>, tensor<10x2x4x11xi1>) -> tensor<10x2x4x11xi1>
    %5 = tosa.minimum %arg2, %arg3 : (tensor<62x87xi32>, tensor<1x1xi32>) -> tensor<62x87xi32>
    return %4, %5 : tensor<10x2x4x11xi1>, tensor<62x87xi32>
  }
}
