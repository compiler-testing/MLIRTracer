module {
  func.func @main(%arg0: tensor<26x17x96x33x90x91xi1>, %arg1: tensor<1x17x96x33x90x1xi1>) -> tensor<26x17x96x33x90x91xi1> {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<26x17x96x33x90x91xi1>, tensor<1x17x96x33x90x1xi1>) -> tensor<26x17x96x33x90x91xi1>
    return %0 : tensor<26x17x96x33x90x91xi1>
  }
}
