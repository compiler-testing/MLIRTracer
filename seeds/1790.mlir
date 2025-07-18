module {
  func.func @main(%arg0: tensor<36xf32>) -> tensor<1xf32> {
    %0 = tosa.rsqrt %arg0 : (tensor<36xf32>) -> tensor<36xf32>
    %1 = tosa.add %0, %0 : (tensor<36xf32>, tensor<36xf32>) -> tensor<36xf32>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<36xf32>) -> tensor<1xf32>
    return %2 : tensor<1xf32>
  }
}
