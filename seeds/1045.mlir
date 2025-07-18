module {
  func.func @main(%arg0: tensor<74x41x68xi1>, %arg1: tensor<1x41x1xi1>) -> tensor<74x41x68xi1> {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<74x41x68xi1>, tensor<1x41x1xi1>) -> tensor<74x41x68xi1>
    %1 = tosa.logical_and %0, %0 : (tensor<74x41x68xi1>, tensor<74x41x68xi1>) -> tensor<74x41x68xi1>
    return %1 : tensor<74x41x68xi1>
  }
}
