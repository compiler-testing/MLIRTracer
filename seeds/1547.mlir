module {
  func.func @main(%arg0: tensor<16x41x21x67xi32>) -> tensor<16x41x21x67xi32> {
    %0 = tosa.clz %arg0 : (tensor<16x41x21x67xi32>) -> tensor<16x41x21x67xi32>
    return %0 : tensor<16x41x21x67xi32>
  }
}
