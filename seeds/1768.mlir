module {
  func.func @main(%arg0: tensor<56x14x58xi1>, %arg1: tensor<81x42x38xf32>) -> (tensor<81x42x38xf32>, tensor<3x1x1xi1>) {
    %r_0 = tosa.const_shape {values = dense<[ 2, 22736, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<56x14x58xi1>, !tosa.shape<3>) -> tensor<2x22736x1xi1>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<2x22736x1xi1>) -> tensor<1x22736x1xi1>
    %2 = tosa.reduce_all %1 {axis = 1 : i32} : (tensor<1x22736x1xi1>) -> tensor<1x1x1xi1>
    %t_3 = tosa.const_shape {values = dense<[ 3, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.tile %2, %t_3 : (tensor<1x1x1xi1>, !tosa.shape<3>) -> tensor<3x2x1xi1>
    %4 = tosa.tanh %arg1 : (tensor<81x42x38xf32>) -> tensor<81x42x38xf32>
    %5 = tosa.reduce_max %3 {axis = 1 : i32} : (tensor<3x2x1xi1>) -> tensor<3x1x1xi1>
    return %4, %5 : tensor<81x42x38xf32>, tensor<3x1x1xi1>
  }
}
