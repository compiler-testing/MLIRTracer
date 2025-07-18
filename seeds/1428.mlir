module {
  func.func @main(%arg0: tensor<63xi32>, %arg1: tensor<63xi32>, %arg2: tensor<30x69x68x19xi1>) -> (tensor<1xi32>, tensor<30x1x68x19xi1>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<63xi32>, tensor<63xi32>) -> tensor<63xi32>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<63xi32>) -> tensor<1xi32>
    %2 = tosa.reduce_any %arg2 {axis = 1 : i32} : (tensor<30x69x68x19xi1>) -> tensor<30x1x68x19xi1>
    return %1, %2 : tensor<1xi32>, tensor<30x1x68x19xi1>
  }
}
