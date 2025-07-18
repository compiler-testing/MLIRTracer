module {
  func.func @main(%arg0: tensor<67x88x77x51x28xf32>, %arg1: tensor<59x40x44x60x89x42xi32>, %arg2: tensor<59x1x44x60x89x1xi32>, %arg3: tensor<41x3x30x87xi1>) -> (tensor<67x88x77x51x28xi1>, tensor<59x40x44x60x89x42xi32>, tensor<1x3x30x87xi1>, tensor<59x40x44x60x89x42xi1>, tensor<1x3x30x87xi1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<67x88x77x51x28xf32>) -> tensor<67x88x77x51x28xf32>
    %1 = tosa.arithmetic_right_shift %arg1, %arg2 {round = false} : (tensor<59x40x44x60x89x42xi32>, tensor<59x1x44x60x89x1xi32>) -> tensor<59x40x44x60x89x42xi32>
    %2 = tosa.greater %0, %0 : (tensor<67x88x77x51x28xf32>, tensor<67x88x77x51x28xf32>) -> tensor<67x88x77x51x28xi1>
    %3 = tosa.maximum %1, %1 : (tensor<59x40x44x60x89x42xi32>, tensor<59x40x44x60x89x42xi32>) -> tensor<59x40x44x60x89x42xi32>
    %4 = tosa.reduce_any %arg3 {axis = 0 : i32} : (tensor<41x3x30x87xi1>) -> tensor<1x3x30x87xi1>
    %5 = tosa.reduce_any %4 {axis = 0 : i32} : (tensor<1x3x30x87xi1>) -> tensor<1x3x30x87xi1>
    %6 = tosa.reverse %4 {axis = 3 : i32} : (tensor<1x3x30x87xi1>) -> tensor<1x3x30x87xi1>
    %7 = tosa.bitwise_and %5, %4 : (tensor<1x3x30x87xi1>, tensor<1x3x30x87xi1>) -> tensor<1x3x30x87xi1>
    %8 = tosa.greater %1, %1 : (tensor<59x40x44x60x89x42xi32>, tensor<59x40x44x60x89x42xi32>) -> tensor<59x40x44x60x89x42xi1>
    %9 = tosa.logical_left_shift %7, %5 : (tensor<1x3x30x87xi1>, tensor<1x3x30x87xi1>) -> tensor<1x3x30x87xi1>
    return %2, %3, %6, %8, %9 : tensor<67x88x77x51x28xi1>, tensor<59x40x44x60x89x42xi32>, tensor<1x3x30x87xi1>, tensor<59x40x44x60x89x42xi1>, tensor<1x3x30x87xi1>
  }
}
