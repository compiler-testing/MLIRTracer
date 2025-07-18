module {
  func.func @main(%arg0: tensor<51x100x24x81xi1>, %arg1: tensor<11x35x8x82x85x85xf32>) -> (tensor<11x35x8x82x85x85xi1>, tensor<11x35x8x82x85x85xf32>, tensor<1x100x24x1xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<51x100x24x81xi1>) -> tensor<1x100x24x81xi1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<1x100x24x81xi1>, tensor<1x100x24x81xi1>) -> tensor<1x100x24x81xi1>
    %2 = tosa.sigmoid %arg1 : (tensor<11x35x8x82x85x85xf32>) -> tensor<11x35x8x82x85x85xf32>
    %3 = tosa.bitwise_or %1, %0 : (tensor<1x100x24x81xi1>, tensor<1x100x24x81xi1>) -> tensor<1x100x24x81xi1>
    %4 = tosa.logical_xor %3, %0 : (tensor<1x100x24x81xi1>, tensor<1x100x24x81xi1>) -> tensor<1x100x24x81xi1>
    %5 = tosa.arithmetic_right_shift %4, %4 {round = false} : (tensor<1x100x24x81xi1>, tensor<1x100x24x81xi1>) -> tensor<1x100x24x81xi1>
    %6 = tosa.equal %2, %2 : (tensor<11x35x8x82x85x85xf32>, tensor<11x35x8x82x85x85xf32>) -> tensor<11x35x8x82x85x85xi1>
    %7 = tosa.floor %2 : (tensor<11x35x8x82x85x85xf32>) -> tensor<11x35x8x82x85x85xf32>
    %8 = tosa.reduce_min %5 {axis = 3 : i32} : (tensor<1x100x24x81xi1>) -> tensor<1x100x24x1xi1>
    return %6, %7, %8 : tensor<11x35x8x82x85x85xi1>, tensor<11x35x8x82x85x85xf32>, tensor<1x100x24x1xi1>
  }
}
