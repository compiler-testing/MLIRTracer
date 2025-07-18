module {
  func.func @main(%arg0: tensor<48xi1>, %arg1: tensor<1xi1>) -> tensor<48xi1> {
    %0 = tosa.sub %arg0, %arg1 : (tensor<48xi1>, tensor<1xi1>) -> tensor<48xi1>
    return %0 : tensor<48xi1>
  }
}
