module {
  func.func @main(%arg0: tensor<76x16x19x90x100x24xi32>, %arg1: tensor<76x1x19x1x1x1xi32>, %arg2: tensor<51xi1>) -> (tensor<76x16x19x90x100x24xi32>, tensor<1xi1>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<76x16x19x90x100x24xi32>, tensor<76x1x19x1x1x1xi32>) -> tensor<76x16x19x90x100x24xi32>
    %1 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<51xi1>) -> tensor<1xi1>
    %2 = tosa.logical_or %1, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %3 = tosa.logical_left_shift %2, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %0, %3 : tensor<76x16x19x90x100x24xi32>, tensor<1xi1>
  }
}
