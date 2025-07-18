module {
  func.func @main(%arg0: tensor<85x5x95xi1>, %arg1: tensor<85x5x1xi1>) -> tensor<85x5x95xi1> {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<85x5x95xi1>, tensor<85x5x1xi1>) -> tensor<85x5x95xi1>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<85x5x95xi1>) -> tensor<85x5x95xi1>
    return %1 : tensor<85x5x95xi1>
  }
}
