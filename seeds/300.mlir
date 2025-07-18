module {
  func.func @main(%arg0: tensor<15xi1>, %arg1: tensor<1xi1>) -> tensor<15xi1> {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<15xi1>, tensor<1xi1>) -> tensor<15xi1>
    return %0 : tensor<15xi1>
  }
}
