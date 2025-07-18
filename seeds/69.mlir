module {
  func.func @main(%arg0: tensor<50x37x99xi1>, %arg1: tensor<1x37x1xi1>) -> tensor<50x37x99xi1> {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<50x37x99xi1>, tensor<1x37x1xi1>) -> tensor<50x37x99xi1>
    return %0 : tensor<50x37x99xi1>
  }
}
