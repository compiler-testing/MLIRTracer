module {
  func.func @main(%arg0: tensor<24x15x51x97xi1>, %arg1: tensor<1x1x51x97xi1>, %arg2: tensor<20x60x78xf32>) -> (tensor<24x1x51x1xi1>, tensor<20x60x78xf32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<24x15x51x97xi1>, tensor<1x1x51x97xi1>) -> tensor<24x15x51x97xi1>
    %1 = tosa.reduce_product %0 {axis = 3 : i32} : (tensor<24x15x51x97xi1>) -> tensor<24x15x51x1xi1>
    %2 = tosa.reduce_max %1 {axis = 1 : i32} : (tensor<24x15x51x1xi1>) -> tensor<24x1x51x1xi1>
    %3 = tosa.logical_or %2, %2 : (tensor<24x1x51x1xi1>, tensor<24x1x51x1xi1>) -> tensor<24x1x51x1xi1>
    %4 = tosa.logical_xor %3, %3 : (tensor<24x1x51x1xi1>, tensor<24x1x51x1xi1>) -> tensor<24x1x51x1xi1>
    %5 = tosa.bitwise_and %4, %3 : (tensor<24x1x51x1xi1>, tensor<24x1x51x1xi1>) -> tensor<24x1x51x1xi1>
    %6 = tosa.tanh %arg2 : (tensor<20x60x78xf32>) -> tensor<20x60x78xf32>
    return %5, %6 : tensor<24x1x51x1xi1>, tensor<20x60x78xf32>
  }
}
