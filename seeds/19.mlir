module {
  func.func @main(%arg0: tensor<64xi32>, %arg1: tensor<1xi32>) -> tensor<64xi1> {
    %0 = tosa.equal %arg0, %arg1 : (tensor<64xi32>, tensor<1xi32>) -> tensor<64xi1>
    return %0 : tensor<64xi1>
  }
}
