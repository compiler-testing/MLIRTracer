module {
  func.func @main(%arg0: tensor<42x30x65x56xi1>, %arg1: tensor<42x1x1x56xi1>) -> tensor<42x30x65x56xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<42x30x65x56xi1>, tensor<42x1x1x56xi1>) -> tensor<42x30x65x56xi1>
    return %0 : tensor<42x30x65x56xi1>
  }
}
