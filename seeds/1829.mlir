module {
  func.func @main(%arg0: tensor<52x74x78x39x5x74xi32>, %arg1: tensor<52x1x78x39x1x1xi32>, %arg2: tensor<80x28x82x78xi1>, %arg3: tensor<7x51x41x92x9x33xf32>, %arg4: tensor<1x1x41x92x1x33xf32>) -> (tensor<52x74x78x39x5x74xi32>, tensor<7x51x41x92x9x33xi1>, tensor<7x51x41x92x9x33xf32>, tensor<80x28x82x78xi1>, tensor<7x51x41x92x9x33xf32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<52x74x78x39x5x74xi32>, tensor<52x1x78x39x1x1xi32>) -> tensor<52x74x78x39x5x74xi32>
    %1 = tosa.logical_not %arg2 : (tensor<80x28x82x78xi1>) -> tensor<80x28x82x78xi1>
    %2 = tosa.pow %arg3, %arg4 : (tensor<7x51x41x92x9x33xf32>, tensor<1x1x41x92x1x33xf32>) -> tensor<7x51x41x92x9x33xf32>
    %3 = tosa.tanh %2 : (tensor<7x51x41x92x9x33xf32>) -> tensor<7x51x41x92x9x33xf32>
    %4 = tosa.greater_equal %3, %3 : (tensor<7x51x41x92x9x33xf32>, tensor<7x51x41x92x9x33xf32>) -> tensor<7x51x41x92x9x33xi1>
    %5 = tosa.maximum %2, %3 : (tensor<7x51x41x92x9x33xf32>, tensor<7x51x41x92x9x33xf32>) -> tensor<7x51x41x92x9x33xf32>
    %6 = tosa.reverse %1 {axis = 3 : i32} : (tensor<80x28x82x78xi1>) -> tensor<80x28x82x78xi1>
    %7 = tosa.ceil %2 : (tensor<7x51x41x92x9x33xf32>) -> tensor<7x51x41x92x9x33xf32>
    return %0, %4, %5, %6, %7 : tensor<52x74x78x39x5x74xi32>, tensor<7x51x41x92x9x33xi1>, tensor<7x51x41x92x9x33xf32>, tensor<80x28x82x78xi1>, tensor<7x51x41x92x9x33xf32>
  }
}
