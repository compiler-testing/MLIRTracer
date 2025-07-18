module {
  func.func @main(%arg0: tensor<96x59x99x46x75xf32>, %arg1: tensor<42x71xi1>, %arg2: tensor<42x71xi1>) -> (tensor<96x59x99x46x75xf32>, tensor<42x71xi1>, tensor<1x71xi1>) {
    %0 = tosa.rsqrt %arg0 : (tensor<96x59x99x46x75xf32>) -> tensor<96x59x99x46x75xf32>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<42x71xi1>, tensor<42x71xi1>) -> tensor<42x71xi1>
    %2 = tosa.logical_xor %1, %1 : (tensor<42x71xi1>, tensor<42x71xi1>) -> tensor<42x71xi1>
    %3 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<42x71xi1>) -> tensor<1x71xi1>
    return %0, %2, %3 : tensor<96x59x99x46x75xf32>, tensor<42x71xi1>, tensor<1x71xi1>
  }
}
