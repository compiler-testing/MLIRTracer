module {
  func.func @main(%arg0: tensor<86x94x42x41xi64>, %arg1: tensor<86x94x73x41xi64>) -> tensor<1x12x6x9xi64> {
    %0 = tosa.concat %arg0, %arg1 {axis = 2 : i32} : (tensor<86x94x42x41xi64>, tensor<86x94x73x41xi64>) -> tensor<86x94x115x41xi64>
    %1 = tosa.clz %0 : (tensor<86x94x115x41xi64>) -> tensor<86x94x115x41xi64>
    %s_2_start = tosa.const_shape {values = dense<[ 3, 82, 52, 32 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_2_size = tosa.const_shape {values = dense<[ 1, 12, 6, 9 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<86x94x115x41xi64>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x12x6x9xi64>
    return %2 : tensor<1x12x6x9xi64>
  }
}
