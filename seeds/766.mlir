module {
  func.func @main(%arg0: tensor<33xi1>, %arg1: tensor<33xi1>, %arg2: tensor<69xf32>, %arg3: tensor<69xf32>) -> (tensor<1xi1>, tensor<1xf32>, tensor<1xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<33xi1>, tensor<33xi1>) -> tensor<33xi1>
    %1 = tosa.pow %arg2, %arg3 : (tensor<69xf32>, tensor<69xf32>) -> tensor<69xf32>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<69xf32>) -> tensor<1xf32>
    %3 = tosa.logical_right_shift %0, %0 : (tensor<33xi1>, tensor<33xi1>) -> tensor<33xi1>
    %4 = tosa.reduce_all %3 {axis = 0 : i32} : (tensor<33xi1>) -> tensor<1xi1>
    %5 = tosa.bitwise_and %4, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.ceil %2 : (tensor<1xf32>) -> tensor<1xf32>
    %7 = tosa.clz %4 : (tensor<1xi1>) -> tensor<1xi1>
    return %5, %6, %7 : tensor<1xi1>, tensor<1xf32>, tensor<1xi1>
  }
}
