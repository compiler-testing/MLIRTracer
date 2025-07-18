module {
  func.func @main(%arg0: tensor<34x18x64x8xf32>) -> tensor<1x18x64x8xf32> {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<34x18x64x8xf32>) -> tensor<1x18x64x8xf32>
    %1 = tosa.abs %0 : (tensor<1x18x64x8xf32>) -> tensor<1x18x64x8xf32>
    %2 = tosa.minimum %1, %1 : (tensor<1x18x64x8xf32>, tensor<1x18x64x8xf32>) -> tensor<1x18x64x8xf32>
    return %2 : tensor<1x18x64x8xf32>
  }
}
