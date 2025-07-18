module {
  func.func @main(%arg0: tensor<3xi1>) -> tensor<3xi1> {
    %0 = tosa.clz %arg0 : (tensor<3xi1>) -> tensor<3xi1>
    return %0 : tensor<3xi1>
  }
}
