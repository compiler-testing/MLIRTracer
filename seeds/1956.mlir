module {
  func.func @main(%arg0: tensor<28xi64>, %arg1: tensor<1xi64>) -> tensor<28xi1> {
    %0 = tosa.equal %arg0, %arg1 : (tensor<28xi64>, tensor<1xi64>) -> tensor<28xi1>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<28xi1>, tensor<28xi1>) -> tensor<28xi1>
    return %1 : tensor<28xi1>
  }
}
