module {
  func.func @main(%arg0: tensor<70x97x89xi1>, %arg1: tensor<6x79xf32>) -> (tensor<1x1x89xi1>, tensor<6x79xi1>, tensor<6x1xf32>) {
    %0 = tosa.reduce_any %arg0 {axis = 1 : i32} : (tensor<70x97x89xi1>) -> tensor<70x1x89xi1>
    %1 = tosa.log %arg1 : (tensor<6x79xf32>) -> tensor<6x79xf32>
    %2 = tosa.clz %0 : (tensor<70x1x89xi1>) -> tensor<70x1x89xi1>
    %3 = tosa.logical_right_shift %2, %0 : (tensor<70x1x89xi1>, tensor<70x1x89xi1>) -> tensor<70x1x89xi1>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<70x1x89xi1>, tensor<70x1x89xi1>) -> tensor<70x1x89xi1>
    %5 = tosa.reduce_all %4 {axis = 0 : i32} : (tensor<70x1x89xi1>) -> tensor<1x1x89xi1>
    %6 = tosa.equal %1, %1 : (tensor<6x79xf32>, tensor<6x79xf32>) -> tensor<6x79xi1>
    %7 = tosa.reduce_max %1 {axis = 1 : i32} : (tensor<6x79xf32>) -> tensor<6x1xf32>
    return %5, %6, %7 : tensor<1x1x89xi1>, tensor<6x79xi1>, tensor<6x1xf32>
  }
}
