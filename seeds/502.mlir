module {
  func.func @main(%arg0: tensor<66xf32>) -> tensor<1xf32> {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<66xf32>) -> tensor<1xf32>
    %1 = tosa.exp %0 : (tensor<1xf32>) -> tensor<1xf32>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %3 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    return %3 : tensor<1xf32>
  }
}
