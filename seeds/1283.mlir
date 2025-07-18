module {
  func.func @main(%arg0: tensor<77x50x18x45xi1>) -> tensor<1x50x18x45xi1> {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<77x50x18x45xi1>) -> tensor<1x50x18x45xi1>
    %1 = tosa.logical_or %0, %0 : (tensor<1x50x18x45xi1>, tensor<1x50x18x45xi1>) -> tensor<1x50x18x45xi1>
    %2 = tosa.arithmetic_right_shift %1, %0 {round = true} : (tensor<1x50x18x45xi1>, tensor<1x50x18x45xi1>) -> tensor<1x50x18x45xi1>
    %3 = tosa.reverse %2 {axis = 0 : i32} : (tensor<1x50x18x45xi1>) -> tensor<1x50x18x45xi1>
    return %3 : tensor<1x50x18x45xi1>
  }
}
