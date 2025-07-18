module {
  func.func @main(%arg0: tensor<18x31x39xf32>, %arg1: tensor<18x75xi1>) -> (tensor<18x1xi1>, tensor<18x31x39xi1>, tensor<18x1xi1>, tensor<18x31x39xf32>, tensor<18x31x39xi1>) {
    %0 = tosa.tanh %arg0 : (tensor<18x31x39xf32>) -> tensor<18x31x39xf32>
    %1 = tosa.pow %0, %0 : (tensor<18x31x39xf32>, tensor<18x31x39xf32>) -> tensor<18x31x39xf32>
    %2 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<18x75xi1>) -> tensor<18x1xi1>
    %3 = tosa.ceil %1 : (tensor<18x31x39xf32>) -> tensor<18x31x39xf32>
    %4 = tosa.arithmetic_right_shift %2, %2 {round = true} : (tensor<18x1xi1>, tensor<18x1xi1>) -> tensor<18x1xi1>
    %5 = tosa.greater %1, %0 : (tensor<18x31x39xf32>, tensor<18x31x39xf32>) -> tensor<18x31x39xi1>
    %6 = tosa.bitwise_xor %2, %2 : (tensor<18x1xi1>, tensor<18x1xi1>) -> tensor<18x1xi1>
    %7 = tosa.reduce_min %6 {axis = 1 : i32} : (tensor<18x1xi1>) -> tensor<18x1xi1>
    %8 = tosa.maximum %0, %0 : (tensor<18x31x39xf32>, tensor<18x31x39xf32>) -> tensor<18x31x39xf32>
    %9 = tosa.greater_equal %0, %3 : (tensor<18x31x39xf32>, tensor<18x31x39xf32>) -> tensor<18x31x39xi1>
    return %4, %5, %7, %8, %9 : tensor<18x1xi1>, tensor<18x31x39xi1>, tensor<18x1xi1>, tensor<18x31x39xf32>, tensor<18x31x39xi1>
  }
}
