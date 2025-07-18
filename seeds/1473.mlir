module {
  func.func @main(%arg0: tensor<37x7x88x58xi32>, %arg1: tensor<1x7x1x58xi32>) -> tensor<37x7x88x58xi1> {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<37x7x88x58xi32>, tensor<1x7x1x58xi32>) -> tensor<37x7x88x58xi32>
    %1 = tosa.greater_equal %0, %0 : (tensor<37x7x88x58xi32>, tensor<37x7x88x58xi32>) -> tensor<37x7x88x58xi1>
    %2 = tosa.add %1, %1 : (tensor<37x7x88x58xi1>, tensor<37x7x88x58xi1>) -> tensor<37x7x88x58xi1>
    return %2 : tensor<37x7x88x58xi1>
  }
}
