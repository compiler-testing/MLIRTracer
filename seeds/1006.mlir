module {
  func.func @main(%arg0: tensor<37x99xi32>, %arg1: tensor<37x1xi32>) -> tensor<37x99xi1> {
    %0 = tosa.equal %arg0, %arg1 : (tensor<37x99xi32>, tensor<37x1xi32>) -> tensor<37x99xi1>
    return %0 : tensor<37x99xi1>
  }
}
