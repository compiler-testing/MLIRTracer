module {
  func.func @main(%arg0: tensor<82xi32>) -> tensor<82xi32> {
    %0 = tosa.bitwise_not %arg0 : (tensor<82xi32>) -> tensor<82xi32>
    %1 = tosa.bitwise_and %0, %0 : (tensor<82xi32>, tensor<82xi32>) -> tensor<82xi32>
    %2 = tosa.bitwise_or %1, %0 : (tensor<82xi32>, tensor<82xi32>) -> tensor<82xi32>
    %3 = tosa.arithmetic_right_shift %2, %1 {round = false} : (tensor<82xi32>, tensor<82xi32>) -> tensor<82xi32>
    return %3 : tensor<82xi32>
  }
}
