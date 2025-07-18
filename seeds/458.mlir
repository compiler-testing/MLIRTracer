module {
  func.func @main(%arg0: tensor<66x98x49x83xi32>, %arg1: tensor<66x98x49x83xi32>) -> tensor<8134xi32> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<66x98x49x83xi32>, tensor<66x98x49x83xi32>) -> tensor<66x98x49x83xi32>
    %1 = tosa.reduce_max %0 {axis = 2 : i32} : (tensor<66x98x49x83xi32>) -> tensor<66x98x1x83xi32>
    %2 = tosa.identity %1 : (tensor<66x98x1x83xi32>) -> tensor<66x98x1x83xi32>
    %3 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<66x98x1x83xi32>) -> tensor<1x98x1x83xi32>
    %4 = tosa.logical_left_shift %3, %3 : (tensor<1x98x1x83xi32>, tensor<1x98x1x83xi32>) -> tensor<1x98x1x83xi32>
    %r_5 = tosa.const_shape {values = dense<[ 8134 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %5 = tosa.reshape %4, %r_5 : (tensor<1x98x1x83xi32>, !tosa.shape<1>) -> tensor<8134xi32>
    return %5 : tensor<8134xi32>
  }
}
