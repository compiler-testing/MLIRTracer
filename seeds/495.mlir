module {
  func.func @main(%arg0: tensor<91xi32>) -> tensor<7xi32> {
    %0 = tosa.clz %arg0 : (tensor<91xi32>) -> tensor<91xi32>
    %1 = tosa.add %0, %0 : (tensor<91xi32>, tensor<91xi32>) -> tensor<91xi32>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<91xi32>) -> tensor<91xi32>
    %3 = tosa.bitwise_not %2 : (tensor<91xi32>) -> tensor<91xi32>
    %s_4_start = tosa.const_shape {values = dense<[ 74 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_4_size = tosa.const_shape {values = dense<[ 7 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.slice %3, %s_4_start, %s_4_size : (tensor<91xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<7xi32>
    %5 = tosa.logical_right_shift %4, %4 : (tensor<7xi32>, tensor<7xi32>) -> tensor<7xi32>
    return %5 : tensor<7xi32>
  }
}
