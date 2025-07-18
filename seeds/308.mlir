module {
  func.func @main(%arg0: tensor<9x41x67x33x76xi1>, %arg1: tensor<9x41x67x33x76xi1>) -> tensor<9x41x67x33x76xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<9x41x67x33x76xi1>, tensor<9x41x67x33x76xi1>) -> tensor<9x41x67x33x76xi1>
    return %0 : tensor<9x41x67x33x76xi1>
  }
}
