module {
  func.func @main(%arg0: tensor<11xi32>, %arg1: tensor<64x93xi1>) -> (tensor<1xi32>, tensor<64x1xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<11xi32>) -> tensor<1xi32>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %3 = tosa.arithmetic_right_shift %2, %2 {round = false} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %4 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<64x93xi1>) -> tensor<64x1xi1>
    return %3, %4 : tensor<1xi32>, tensor<64x1xi1>
  }
}
