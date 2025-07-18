module {
  func.func @main(%arg0: tensor<49x46xi1>, %arg1: tensor<1x46xi1>) -> tensor<49x46xi1> {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<49x46xi1>, tensor<1x46xi1>) -> tensor<49x46xi1>
    return %0 : tensor<49x46xi1>
  }
}
