module {
  func.func @main(%arg0: tensor<24x33x70xi1>, %arg1: tensor<41x90xf32>, %arg2: tensor<41x90xf32>, %arg3: tensor<89x70xi32>, %arg4: tensor<89x1xi32>) -> (tensor<41x90xf32>, tensor<89x70xi32>, tensor<1x33x70xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<24x33x70xi1>) -> tensor<1x33x70xi1>
    %1 = tosa.pow %arg1, %arg2 : (tensor<41x90xf32>, tensor<41x90xf32>) -> tensor<41x90xf32>
    %2 = tosa.logical_xor %0, %0 : (tensor<1x33x70xi1>, tensor<1x33x70xi1>) -> tensor<1x33x70xi1>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<1x33x70xi1>, tensor<1x33x70xi1>) -> tensor<1x33x70xi1>
    %4 = tosa.logical_and %3, %0 : (tensor<1x33x70xi1>, tensor<1x33x70xi1>) -> tensor<1x33x70xi1>
    %5 = tosa.intdiv %arg3, %arg4 : (tensor<89x70xi32>, tensor<89x1xi32>) -> tensor<89x70xi32>
    %6 = tosa.reduce_product %4 {axis = 0 : i32} : (tensor<1x33x70xi1>) -> tensor<1x33x70xi1>
    %7 = tosa.bitwise_or %5, %5 : (tensor<89x70xi32>, tensor<89x70xi32>) -> tensor<89x70xi32>
    %8 = tosa.reduce_all %6 {axis = 0 : i32} : (tensor<1x33x70xi1>) -> tensor<1x33x70xi1>
    return %1, %7, %8 : tensor<41x90xf32>, tensor<89x70xi32>, tensor<1x33x70xi1>
  }
}
