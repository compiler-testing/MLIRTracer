module {
  func.func @main(%arg0: tensor<65x36x22x1x14xi1>) -> tensor<65x36x22x1x14xi1> {
    %0 = tosa.identity %arg0 : (tensor<65x36x22x1x14xi1>) -> tensor<65x36x22x1x14xi1>
    return %0 : tensor<65x36x22x1x14xi1>
  }
}
