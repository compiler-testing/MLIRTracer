module {
  func.func @main(%arg0: tensor<51xi64>, %arg1: tensor<1xi64>) -> tensor<51xi64> {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<51xi64>, tensor<1xi64>) -> tensor<51xi64>
    return %0 : tensor<51xi64>
  }
}
