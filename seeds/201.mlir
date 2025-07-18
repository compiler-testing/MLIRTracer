module {
  func.func @main(%arg0: tensor<84x59x95x69xi1>) -> tensor<84x1x95x69xi1> {
    %0 = tosa.reduce_any %arg0 {axis = 1 : i32} : (tensor<84x59x95x69xi1>) -> tensor<84x1x95x69xi1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<84x1x95x69xi1>, tensor<84x1x95x69xi1>) -> tensor<84x1x95x69xi1>
    %2 = tosa.logical_or %1, %1 : (tensor<84x1x95x69xi1>, tensor<84x1x95x69xi1>) -> tensor<84x1x95x69xi1>
    return %2 : tensor<84x1x95x69xi1>
  }
}
