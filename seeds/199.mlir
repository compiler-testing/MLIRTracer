module {
  func.func @main(%arg0: tensor<35x47xf32>) -> tensor<35x1xf32> {
    %0 = tosa.reciprocal %arg0 : (tensor<35x47xf32>) -> tensor<35x47xf32>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<35x47xf32>) -> tensor<35x1xf32>
    %2 = tosa.maximum %1, %1 : (tensor<35x1xf32>, tensor<35x1xf32>) -> tensor<35x1xf32>
    %3 = tosa.minimum %2, %2 : (tensor<35x1xf32>, tensor<35x1xf32>) -> tensor<35x1xf32>
    return %3 : tensor<35x1xf32>
  }
}
