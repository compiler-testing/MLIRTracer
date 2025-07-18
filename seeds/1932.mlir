module {
  func.func @main(%arg0: tensor<88xi1>, %arg1: tensor<88xi1>) -> tensor<88xi1> {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<88xi1>, tensor<88xi1>) -> tensor<88xi1>
    %1 = tosa.bitwise_and %0, %0 : (tensor<88xi1>, tensor<88xi1>) -> tensor<88xi1>
    return %1 : tensor<88xi1>
  }
}
