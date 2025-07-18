module {
  func.func @main(%arg0: tensor<54x8x45xi32>, %arg1: tensor<1x1x1xi32>) -> tensor<54x8x1xi32> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<54x8x45xi32>, tensor<1x1x1xi32>) -> tensor<54x8x45xi32>
    %1 = tosa.reduce_max %0 {axis = 2 : i32} : (tensor<54x8x45xi32>) -> tensor<54x8x1xi32>
    return %1 : tensor<54x8x1xi32>
  }
}
