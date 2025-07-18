module {
  func.func @main(%arg0: tensor<58x80xi1>, %arg1: tensor<1x1xi1>, %arg2: tensor<13xi32>, %arg3: tensor<1xi32>) -> (tensor<58x80xi1>, tensor<1xi32>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<58x80xi1>, tensor<1x1xi1>) -> tensor<58x80xi1>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<58x80xi1>, tensor<58x80xi1>) -> tensor<58x80xi1>
    %2 = tosa.intdiv %arg2, %arg3 : (tensor<13xi32>, tensor<1xi32>) -> tensor<13xi32>
    %3 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<13xi32>) -> tensor<1xi32>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %5 = tosa.arithmetic_right_shift %4, %3 {round = false} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    return %1, %5 : tensor<58x80xi1>, tensor<1xi32>
  }
}
