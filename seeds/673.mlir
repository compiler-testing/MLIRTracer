module {
  func.func @main(%arg0: tensor<13x74x48x73xf32>, %arg1: tensor<62xi8>, %arg2: tensor<1xi8>) -> (tensor<8x8x2x7xf32>, tensor<i32>, tensor<1xi8>, tensor<62xi8>) {
    %0 = tosa.reduce_min %arg0 {axis = 3 : i32} : (tensor<13x74x48x73xf32>) -> tensor<13x74x48x1xf32>
    %s_1_start = tosa.const_shape {values = dense<[ 5, 1, 11, 0 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_1_size = tosa.const_shape {values = dense<[ 8, 8, 2, 7 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<13x74x48x1xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<8x8x2x7xf32>
    %2 = tosa.bitwise_or %arg1, %arg2 : (tensor<62xi8>, tensor<1xi8>) -> tensor<62xi8>
    %3 = tosa.bitwise_not %2 : (tensor<62xi8>) -> tensor<62xi8>
    %4 = tosa.argmax %3 {axis = 0 : i32} : (tensor<62xi8>) -> tensor<i32>
    %5 = tosa.bitwise_and %4, %4 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = tosa.bitwise_not %5 : (tensor<i32>) -> tensor<i32>
    %7 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<62xi8>) -> tensor<1xi8>
    %8 = tosa.bitwise_not %2 : (tensor<62xi8>) -> tensor<62xi8>
    return %1, %6, %7, %8 : tensor<8x8x2x7xf32>, tensor<i32>, tensor<1xi8>, tensor<62xi8>
  }
}
