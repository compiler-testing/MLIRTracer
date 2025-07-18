module {
  func.func @main(%arg0: tensor<95x13x21x73x18x46xi64>, %arg1: tensor<73x17x6xi1>, %arg2: tensor<46x65xf32>) -> (tensor<4x8x8x10x9x6xi64>, tensor<138x65xf32>, tensor<1x1x6xi1>) {
    %0 = tosa.abs %arg0 : (tensor<95x13x21x73x18x46xi64>) -> tensor<95x13x21x73x18x46xi64>
    %1 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<73x17x6xi1>) -> tensor<73x1x6xi1>
    %2 = tosa.reciprocal %arg2 : (tensor<46x65xf32>) -> tensor<46x65xf32>
    %3 = tosa.abs %0 : (tensor<95x13x21x73x18x46xi64>) -> tensor<95x13x21x73x18x46xi64>
    %4 = tosa.bitwise_not %1 : (tensor<73x1x6xi1>) -> tensor<73x1x6xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 39, 5, 13, 60, 9, 33 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_5_size = tosa.const_shape {values = dense<[ 4, 8, 8, 10, 9, 6 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %5 = tosa.slice %3, %s_5_start, %s_5_size : (tensor<95x13x21x73x18x46xi64>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<4x8x8x10x9x6xi64>
    %t_6 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.tile %2, %t_6 : (tensor<46x65xf32>, !tosa.shape<2>) -> tensor<138x65xf32>
    %7 = tosa.reduce_all %4 {axis = 0 : i32} : (tensor<73x1x6xi1>) -> tensor<1x1x6xi1>
    return %5, %6, %7 : tensor<4x8x8x10x9x6xi64>, tensor<138x65xf32>, tensor<1x1x6xi1>
  }
}
