module {
  func.func @main(%arg0: tensor<11x58x50x79x79xi64>) -> tensor<11x58x50x79x79xi64> {
    %0 = tosa.clz %arg0 : (tensor<11x58x50x79x79xi64>) -> tensor<11x58x50x79x79xi64>
    return %0 : tensor<11x58x50x79x79xi64>
  }
}
