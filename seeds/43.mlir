module {
  func.func @main(%arg0: tensor<14x98xf32>) -> tensor<14x1xf32> {
    %0 = tosa.reciprocal %arg0 : (tensor<14x98xf32>) -> tensor<14x98xf32>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<14x98xf32>) -> tensor<14x1xf32>
    return %1 : tensor<14x1xf32>
  }
}
