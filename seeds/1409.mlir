module {
  func.func @main(%arg0: tensor<60x34x61x61x35x67xf32>, %arg1: tensor<i1>, %arg2: tensor<i1>, %arg3: tensor<29x59x64x82xi32>) -> (tensor<i1>, tensor<3x12x1x1x6x3xf32>, tensor<29x59x1x82xi32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<60x34x61x61x35x67xf32>) -> tensor<60x34x61x61x35x67xf32>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %s_2_start = tosa.const_shape {values = dense<[ 28, 16, 17, 30, 29, 10 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_2_size = tosa.const_shape {values = dense<[ 3, 12, 1, 1, 6, 3 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %2 = tosa.slice %0, %s_2_start, %s_2_size : (tensor<60x34x61x61x35x67xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<3x12x1x1x6x3xf32>
    %3 = tosa.reduce_max %arg3 {axis = 2 : i32} : (tensor<29x59x64x82xi32>) -> tensor<29x59x1x82xi32>
    return %1, %2, %3 : tensor<i1>, tensor<3x12x1x1x6x3xf32>, tensor<29x59x1x82xi32>
  }
}
