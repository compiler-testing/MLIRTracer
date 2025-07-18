module {
  func.func @main(%arg0: tensor<78x20x81xi32>) -> tensor<78x20x81xi1> {
    %0 = tosa.abs %arg0 : (tensor<78x20x81xi32>) -> tensor<78x20x81xi32>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<78x20x81xi32>, tensor<78x20x81xi32>) -> tensor<78x20x81xi32>
    %2 = tosa.greater_equal %1, %0 : (tensor<78x20x81xi32>, tensor<78x20x81xi32>) -> tensor<78x20x81xi1>
    %3 = tosa.logical_or %2, %2 : (tensor<78x20x81xi1>, tensor<78x20x81xi1>) -> tensor<78x20x81xi1>
    return %3 : tensor<78x20x81xi1>
  }
}
