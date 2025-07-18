module {
  func.func @main(%arg0: tensor<47x2xi32>, %arg1: tensor<1x2xi32>, %arg2: tensor<88xi32>, %arg3: tensor<88xi32>) -> (tensor<1x2xi1>, tensor<1xi32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<47x2xi32>, tensor<1x2xi32>) -> tensor<47x2xi1>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<47x2xi1>, tensor<47x2xi1>) -> tensor<47x2xi1>
    %2 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<47x2xi1>, tensor<47x2xi1>) -> tensor<47x2xi1>
    %3 = tosa.reduce_any %2 {axis = 0 : i32} : (tensor<47x2xi1>) -> tensor<1x2xi1>
    %4 = tosa.minimum %arg2, %arg3 : (tensor<88xi32>, tensor<88xi32>) -> tensor<88xi32>
    %5 = tosa.clz %3 : (tensor<1x2xi1>) -> tensor<1x2xi1>
    %6 = tosa.reduce_max %4 {axis = 0 : i32} : (tensor<88xi32>) -> tensor<1xi32>
    %7 = tosa.logical_or %5, %5 : (tensor<1x2xi1>, tensor<1x2xi1>) -> tensor<1x2xi1>
    %8 = tosa.bitwise_xor %6, %6 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    return %7, %8 : tensor<1x2xi1>, tensor<1xi32>
  }
}
