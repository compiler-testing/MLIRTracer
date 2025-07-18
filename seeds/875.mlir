module {
  func.func @main(%arg0: tensor<38x34x96xi1>, %arg1: tensor<1x1x1xi1>) -> tensor<38x34x96xi1> {
    %0 = tosa.add %arg0, %arg1 : (tensor<38x34x96xi1>, tensor<1x1x1xi1>) -> tensor<38x34x96xi1>
    return %0 : tensor<38x34x96xi1>
  }
}
