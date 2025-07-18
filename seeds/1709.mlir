module {
  func.func @main(%arg0: tensor<74x31x30x26x69xi1>) -> tensor<74x31x30x26x69xi1> {
    %0 = tosa.bitwise_not %arg0 : (tensor<74x31x30x26x69xi1>) -> tensor<74x31x30x26x69xi1>
    return %0 : tensor<74x31x30x26x69xi1>
  }
}
