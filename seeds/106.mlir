module {
  func.func @main(%arg0: tensor<15x34x80x49x47xi1>, %arg1: tensor<15x1x1x49x1xi1>) -> tensor<15x34x80x49x47xi1> {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<15x34x80x49x47xi1>, tensor<15x1x1x49x1xi1>) -> tensor<15x34x80x49x47xi1>
    return %0 : tensor<15x34x80x49x47xi1>
  }
}
