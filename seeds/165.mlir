module {
  func.func @main(%arg0: tensor<14x2x30x57xf32>) -> tensor<42x2x60x3xf32> {
    %0 = tosa.reduce_max %arg0 {axis = 3 : i32} : (tensor<14x2x30x57xf32>) -> tensor<14x2x30x1xf32>
    %1 = tosa.log %0 : (tensor<14x2x30x1xf32>) -> tensor<14x2x30x1xf32>
    %t_2 = tosa.const_shape {values = dense<[ 3, 1, 2, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.tile %1, %t_2 : (tensor<14x2x30x1xf32>, !tosa.shape<4>) -> tensor<42x2x60x3xf32>
    return %2 : tensor<42x2x60x3xf32>
  }
}
