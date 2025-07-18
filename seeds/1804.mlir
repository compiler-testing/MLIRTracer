module {
  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
    %0 = tosa.clz %arg0 : (tensor<i64>) -> tensor<i64>
    return %0 : tensor<i64>
  }
}
