module {
  func.func @main(%arg0: tensor<78x27x35xi1>, %arg1: tensor<65x93x8x18x75x74xi64>, %arg2: tensor<65x1x8x18x1x1xi64>) -> (tensor<65x93x8x18x75x74xi1>, tensor<1x10x12xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<78x27x35xi1>) -> tensor<78x1x35xi1>
    %1 = tosa.greater_equal %arg1, %arg2 : (tensor<65x93x8x18x75x74xi64>, tensor<65x1x8x18x1x1xi64>) -> tensor<65x93x8x18x75x74xi1>
    %2 = tosa.clz %0 : (tensor<78x1x35xi1>) -> tensor<78x1x35xi1>
    %s_3_start = tosa.const_shape {values = dense<[ 70, 0, 23 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_3_size = tosa.const_shape {values = dense<[ 3, 10, 12 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<78x1x35xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<3x10x12xi1>
    %4 = tosa.reduce_any %3 {axis = 0 : i32} : (tensor<3x10x12xi1>) -> tensor<1x10x12xi1>
    %5 = tosa.reverse %4 {axis = 0 : i32} : (tensor<1x10x12xi1>) -> tensor<1x10x12xi1>
    return %1, %5 : tensor<65x93x8x18x75x74xi1>, tensor<1x10x12xi1>
  }
}
