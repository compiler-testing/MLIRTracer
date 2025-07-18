module {
  func.func @main(%arg0: tensor<92x100x13x23xi32>) -> tensor<552x6x26x2xi32> {
    %0 = tosa.reduce_max %arg0 {axis = 3 : i32} : (tensor<92x100x13x23xi32>) -> tensor<92x100x13x1xi32>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<92x100x13x1xi32>) -> tensor<92x1x13x1xi32>
    %2 = tosa.add %1, %1 : (tensor<92x1x13x1xi32>, tensor<92x1x13x1xi32>) -> tensor<92x1x13x1xi32>
    %t_3 = tosa.const_shape {values = dense<[ 2, 3, 2, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.tile %2, %t_3 : (tensor<92x1x13x1xi32>, !tosa.shape<4>) -> tensor<184x3x26x2xi32>
    %t_4 = tosa.const_shape {values = dense<[ 3, 2, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = tosa.tile %3, %t_4 : (tensor<184x3x26x2xi32>, !tosa.shape<4>) -> tensor<552x6x26x2xi32>
    return %4 : tensor<552x6x26x2xi32>
  }
}
