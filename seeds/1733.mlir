module {
  func.func @main(%arg0: tensor<28x79x46x16xf32>, %arg1: tensor<77x91xi32>, %arg2: tensor<77x91xi32>) -> (tensor<28x79x46x16xf32>, tensor<28x79x46x16xi1>, tensor<28x79x46x1xi1>, tensor<77x91xi32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<28x79x46x16xf32>) -> tensor<28x79x46x16xf32>
    %1 = tosa.greater %0, %0 : (tensor<28x79x46x16xf32>, tensor<28x79x46x16xf32>) -> tensor<28x79x46x16xi1>
    %2 = tosa.pow %0, %0 : (tensor<28x79x46x16xf32>, tensor<28x79x46x16xf32>) -> tensor<28x79x46x16xf32>
    %3 = tosa.arithmetic_right_shift %1, %1 {round = false} : (tensor<28x79x46x16xi1>, tensor<28x79x46x16xi1>) -> tensor<28x79x46x16xi1>
    %4 = tosa.identity %3 : (tensor<28x79x46x16xi1>) -> tensor<28x79x46x16xi1>
    %5 = tosa.reduce_sum %4 {axis = 3 : i32} : (tensor<28x79x46x16xi1>) -> tensor<28x79x46x1xi1>
    %6 = tosa.greater_equal %0, %0 : (tensor<28x79x46x16xf32>, tensor<28x79x46x16xf32>) -> tensor<28x79x46x16xi1>
    %7 = tosa.bitwise_xor %5, %5 : (tensor<28x79x46x1xi1>, tensor<28x79x46x1xi1>) -> tensor<28x79x46x1xi1>
    %8 = tosa.arithmetic_right_shift %7, %7 {round = false} : (tensor<28x79x46x1xi1>, tensor<28x79x46x1xi1>) -> tensor<28x79x46x1xi1>
    %9 = tosa.intdiv %arg1, %arg2 : (tensor<77x91xi32>, tensor<77x91xi32>) -> tensor<77x91xi32>
    return %2, %6, %8, %9 : tensor<28x79x46x16xf32>, tensor<28x79x46x16xi1>, tensor<28x79x46x1xi1>, tensor<77x91xi32>
  }
}
