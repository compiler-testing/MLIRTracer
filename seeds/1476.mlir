module {
  func.func @main(%arg0: tensor<20x70x28xf32>) -> (tensor<20x1x28xf32>, tensor<20x70x28xf32>, tensor<20x1x28xi1>) {
    %0 = tosa.floor %arg0 : (tensor<20x70x28xf32>) -> tensor<20x70x28xf32>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<20x70x28xf32>) -> tensor<20x1x28xf32>
    %2 = tosa.greater %1, %1 : (tensor<20x1x28xf32>, tensor<20x1x28xf32>) -> tensor<20x1x28xi1>
    %3 = tosa.maximum %1, %1 : (tensor<20x1x28xf32>, tensor<20x1x28xf32>) -> tensor<20x1x28xf32>
    %4 = tosa.bitwise_or %2, %2 : (tensor<20x1x28xi1>, tensor<20x1x28xi1>) -> tensor<20x1x28xi1>
    %5 = tosa.sigmoid %0 : (tensor<20x70x28xf32>) -> tensor<20x70x28xf32>
    %6 = tosa.arithmetic_right_shift %4, %4 {round = true} : (tensor<20x1x28xi1>, tensor<20x1x28xi1>) -> tensor<20x1x28xi1>
    return %3, %5, %6 : tensor<20x1x28xf32>, tensor<20x70x28xf32>, tensor<20x1x28xi1>
  }
}
