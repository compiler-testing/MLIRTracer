module {
  func.func @main(%arg0: tensor<20x1x87xi1>, %arg1: tensor<15x29xi32>, %arg2: tensor<15x29xi32>) -> (tensor<1x1x1xi1>, tensor<15x29xi32>) {
    %0 = tosa.reduce_all %arg0 {axis = 2 : i32} : (tensor<20x1x87xi1>) -> tensor<20x1x1xi1>
    %1 = tosa.abs %0 : (tensor<20x1x1xi1>) -> tensor<20x1x1xi1>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<20x1x1xi1>) -> tensor<20x1x1xi1>
    %3 = tosa.intdiv %arg1, %arg2 : (tensor<15x29xi32>, tensor<15x29xi32>) -> tensor<15x29xi32>
    %4 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<20x1x1xi1>) -> tensor<1x1x1xi1>
    %5 = tosa.arithmetic_right_shift %4, %4 {round = true} : (tensor<1x1x1xi1>, tensor<1x1x1xi1>) -> tensor<1x1x1xi1>
    %6 = tosa.add %3, %3 : (tensor<15x29xi32>, tensor<15x29xi32>) -> tensor<15x29xi32>
    return %5, %6 : tensor<1x1x1xi1>, tensor<15x29xi32>
  }
}
