module {
  func.func @main(%arg0: tensor<45x74x79xf32>, %arg1: tensor<45x12x81x64x1xi1>, %arg2: tensor<45x1x81x1x1xi1>) -> (tensor<45x74x79xf32>, tensor<45x1x79xf32>, tensor<45x74x79xf32>, tensor<1x72x6x6480xi1>) {
    %0 = tosa.reverse %arg0 {axis = 2 : i32} : (tensor<45x74x79xf32>) -> tensor<45x74x79xf32>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<45x12x81x64x1xi1>, tensor<45x1x81x1x1xi1>) -> tensor<45x12x81x64x1xi1>
    %2 = tosa.logical_or %1, %1 : (tensor<45x12x81x64x1xi1>, tensor<45x12x81x64x1xi1>) -> tensor<45x12x81x64x1xi1>
    %3 = tosa.exp %0 : (tensor<45x74x79xf32>) -> tensor<45x74x79xf32>
    %4 = tosa.logical_or %2, %1 : (tensor<45x12x81x64x1xi1>, tensor<45x12x81x64x1xi1>) -> tensor<45x12x81x64x1xi1>
    %5 = tosa.reverse %3 {axis = 0 : i32} : (tensor<45x74x79xf32>) -> tensor<45x74x79xf32>
    %6 = tosa.reduce_max %3 {axis = 1 : i32} : (tensor<45x74x79xf32>) -> tensor<45x1x79xf32>
    %7 = tosa.exp %0 : (tensor<45x74x79xf32>) -> tensor<45x74x79xf32>
    %r_8 = tosa.const_shape {values = dense<[ 1, 72, 6, 6480 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %8 = tosa.reshape %4, %r_8 : (tensor<45x12x81x64x1xi1>, !tosa.shape<4>) -> tensor<1x72x6x6480xi1>
    return %5, %6, %7, %8 : tensor<45x74x79xf32>, tensor<45x1x79xf32>, tensor<45x74x79xf32>, tensor<1x72x6x6480xi1>
  }
}
