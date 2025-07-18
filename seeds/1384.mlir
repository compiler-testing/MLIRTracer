module {
  func.func @main(%arg0: tensor<4xi1>, %arg1: tensor<8xf32>) -> (tensor<1xi1>, tensor<8xf32>, tensor<1xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<4xi1>) -> tensor<1xi1>
    %1 = tosa.identity %0 : (tensor<1xi1>) -> tensor<1xi1>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %3 = tosa.logical_left_shift %2, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.logical_left_shift %3, %3 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.bitwise_xor %4, %3 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.tanh %arg1 : (tensor<8xf32>) -> tensor<8xf32>
    %7 = tosa.bitwise_not %5 : (tensor<1xi1>) -> tensor<1xi1>
    %8 = tosa.minimum %6, %6 : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %9 = tosa.bitwise_not %3 : (tensor<1xi1>) -> tensor<1xi1>
    return %7, %8, %9 : tensor<1xi1>, tensor<8xf32>, tensor<1xi1>
  }
}
