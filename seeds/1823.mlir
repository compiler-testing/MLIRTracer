module {
  func.func @main(%arg0: tensor<27x85x14x29xi1>, %arg1: tensor<1x1x14x29xi1>, %arg2: tensor<81x59x37x100x72x52xi32>, %arg3: tensor<81x1x37x1x1x1xi32>, %arg4: tensor<60x98x87x22x2x63xf32>) -> (tensor<1x1x14x29xi1>, tensor<60x98x87x22x2x63xf32>, tensor<3x8x1x4x5x12xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<27x85x14x29xi1>, tensor<1x1x14x29xi1>) -> tensor<27x85x14x29xi1>
    %1 = tosa.logical_or %0, %0 : (tensor<27x85x14x29xi1>, tensor<27x85x14x29xi1>) -> tensor<27x85x14x29xi1>
    %2 = tosa.reduce_any %1 {axis = 1 : i32} : (tensor<27x85x14x29xi1>) -> tensor<27x1x14x29xi1>
    %3 = tosa.intdiv %arg2, %arg3 : (tensor<81x59x37x100x72x52xi32>, tensor<81x1x37x1x1x1xi32>) -> tensor<81x59x37x100x72x52xi32>
    %s_4_start = tosa.const_shape {values = dense<[ 46, 51, 15, 17, 57, 24 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_4_size = tosa.const_shape {values = dense<[ 3, 8, 1, 4, 5, 12 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %4 = tosa.slice %3, %s_4_start, %s_4_size : (tensor<81x59x37x100x72x52xi32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<3x8x1x4x5x12xi32>
    %5 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<27x1x14x29xi1>) -> tensor<1x1x14x29xi1>
    %6 = tosa.add %4, %4 : (tensor<3x8x1x4x5x12xi32>, tensor<3x8x1x4x5x12xi32>) -> tensor<3x8x1x4x5x12xi32>
    %7 = tosa.add %5, %5 : (tensor<1x1x14x29xi1>, tensor<1x1x14x29xi1>) -> tensor<1x1x14x29xi1>
    %8 = tosa.logical_or %7, %7 : (tensor<1x1x14x29xi1>, tensor<1x1x14x29xi1>) -> tensor<1x1x14x29xi1>
    %9 = tosa.bitwise_or %8, %8 : (tensor<1x1x14x29xi1>, tensor<1x1x14x29xi1>) -> tensor<1x1x14x29xi1>
    %10 = tosa.reciprocal %arg4 : (tensor<60x98x87x22x2x63xf32>) -> tensor<60x98x87x22x2x63xf32>
    %11 = tosa.equal %6, %4 : (tensor<3x8x1x4x5x12xi32>, tensor<3x8x1x4x5x12xi32>) -> tensor<3x8x1x4x5x12xi1>
    %12 = tosa.log %10 : (tensor<60x98x87x22x2x63xf32>) -> tensor<60x98x87x22x2x63xf32>
    %13 = tosa.bitwise_not %11 : (tensor<3x8x1x4x5x12xi1>) -> tensor<3x8x1x4x5x12xi1>
    return %9, %12, %13 : tensor<1x1x14x29xi1>, tensor<60x98x87x22x2x63xf32>, tensor<3x8x1x4x5x12xi1>
  }
}
