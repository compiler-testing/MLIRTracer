module {
  func.func @main(%arg0: tensor<87xf32>, %arg1: tensor<19x59x32x37x8x14xi1>, %arg2: tensor<19x59x32x37x1x1xi1>) -> (tensor<1xf32>, tensor<19x59x32x37x8x14xi1>, tensor<19x59x32x37x8x14xi1>) {
    %0 = tosa.log %arg0 : (tensor<87xf32>) -> tensor<87xf32>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<19x59x32x37x8x14xi1>, tensor<19x59x32x37x1x1xi1>) -> tensor<19x59x32x37x8x14xi1>
    %2 = tosa.logical_and %1, %1 : (tensor<19x59x32x37x8x14xi1>, tensor<19x59x32x37x8x14xi1>) -> tensor<19x59x32x37x8x14xi1>
    %3 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<87xf32>) -> tensor<1xf32>
    %4 = tosa.logical_xor %2, %1 : (tensor<19x59x32x37x8x14xi1>, tensor<19x59x32x37x8x14xi1>) -> tensor<19x59x32x37x8x14xi1>
    %5 = tosa.bitwise_xor %1, %1 : (tensor<19x59x32x37x8x14xi1>, tensor<19x59x32x37x8x14xi1>) -> tensor<19x59x32x37x8x14xi1>
    return %3, %4, %5 : tensor<1xf32>, tensor<19x59x32x37x8x14xi1>, tensor<19x59x32x37x8x14xi1>
  }
}
