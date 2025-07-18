module {
  func.func @main(%arg0: tensor<7x96x60xf32>, %arg1: tensor<55x24x10x13x89xi64>) -> (tensor<55x24x10x13x89xi64>, tensor<7x1x60xf32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<7x96x60xf32>) -> tensor<7x96x60xf32>
    %1 = tosa.bitwise_not %arg1 : (tensor<55x24x10x13x89xi64>) -> tensor<55x24x10x13x89xi64>
    %2 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<7x96x60xf32>) -> tensor<7x1x60xf32>
    return %1, %2 : tensor<55x24x10x13x89xi64>, tensor<7x1x60xf32>
  }
}
