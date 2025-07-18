module {
  func.func @main(%arg0: tensor<10xi1>, %arg1: tensor<1xi1>, %arg2: tensor<47x68x76x96x17x92xf32>) -> (tensor<10xi1>, tensor<47x68x76x96x17x92xi1>, tensor<47x68x76x96x17x92xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<10xi1>, tensor<1xi1>) -> tensor<10xi1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
    %2 = tosa.floor %arg2 : (tensor<47x68x76x96x17x92xf32>) -> tensor<47x68x76x96x17x92xf32>
    %3 = tosa.bitwise_not %1 : (tensor<10xi1>) -> tensor<10xi1>
    %4 = tosa.add %2, %2 : (tensor<47x68x76x96x17x92xf32>, tensor<47x68x76x96x17x92xf32>) -> tensor<47x68x76x96x17x92xf32>
    %5 = tosa.sub %4, %2 : (tensor<47x68x76x96x17x92xf32>, tensor<47x68x76x96x17x92xf32>) -> tensor<47x68x76x96x17x92xf32>
    %6 = tosa.equal %5, %2 : (tensor<47x68x76x96x17x92xf32>, tensor<47x68x76x96x17x92xf32>) -> tensor<47x68x76x96x17x92xi1>
    %7 = tosa.exp %5 : (tensor<47x68x76x96x17x92xf32>) -> tensor<47x68x76x96x17x92xf32>
    return %3, %6, %7 : tensor<10xi1>, tensor<47x68x76x96x17x92xi1>, tensor<47x68x76x96x17x92xf32>
  }
}
