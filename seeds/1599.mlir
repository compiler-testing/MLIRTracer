module {
  func.func @main(%arg0: tensor<72x21xi64>) -> tensor<1x21xi64> {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<72x21xi64>) -> tensor<1x21xi64>
    return %0 : tensor<1x21xi64>
  }
}
