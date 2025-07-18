module {
  func.func @main(%arg0: tensor<5x15x9x5x65x97xi32>) -> tensor<5x15x9x5x65x97xi32> {
    %0 = tosa.clz %arg0 : (tensor<5x15x9x5x65x97xi32>) -> tensor<5x15x9x5x65x97xi32>
    return %0 : tensor<5x15x9x5x65x97xi32>
  }
}
