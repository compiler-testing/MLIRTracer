module {
  func.func @main(%arg0: tensor<47xi32>, %arg1: tensor<47xi32>) -> tensor<47xi1> {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<47xi32>, tensor<47xi32>) -> tensor<47xi32>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<47xi32>, tensor<47xi32>) -> tensor<47xi32>
    %2 = tosa.greater %1, %1 : (tensor<47xi32>, tensor<47xi32>) -> tensor<47xi1>
    return %2 : tensor<47xi1>
  }
}
