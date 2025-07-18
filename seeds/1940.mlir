module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<21x17x24xi64>, %arg2: tensor<1x1x1xi64>) -> (tensor<i8>, tensor<42x17x24xi64>, tensor<1x8x6xi64>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<i8>) -> tensor<i8>
    %1 = tosa.maximum %arg1, %arg2 : (tensor<21x17x24xi64>, tensor<1x1x1xi64>) -> tensor<21x17x24xi64>
    %2 = tosa.reverse %1 {axis = 2 : i32} : (tensor<21x17x24xi64>) -> tensor<21x17x24xi64>
    %3 = tosa.concat %1, %1 {axis = 0 : i32} : (tensor<21x17x24xi64>, tensor<21x17x24xi64>) -> tensor<42x17x24xi64>
    %s_4_start = tosa.const_shape {values = dense<[ 6, 9, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_4_size = tosa.const_shape {values = dense<[ 11, 2, 4 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.slice %2, %s_4_start, %s_4_size : (tensor<21x17x24xi64>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<11x2x4xi64>
    %s_5_start = tosa.const_shape {values = dense<[ 3, 0, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_5_size = tosa.const_shape {values = dense<[ 1, 8, 6 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<11x2x4xi64>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x8x6xi64>
    return %0, %3, %5 : tensor<i8>, tensor<42x17x24xi64>, tensor<1x8x6xi64>
  }
}
