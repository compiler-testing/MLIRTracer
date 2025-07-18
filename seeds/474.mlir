module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<71xi1>) -> (tensor<i1>, tensor<1xi1>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<71xi1>) -> tensor<1xi1>
    %3 = tosa.bitwise_not %1 : (tensor<i1>) -> tensor<i1>
    %4 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.bitwise_or %4, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %3, %5 : tensor<i1>, tensor<1xi1>
  }
}
