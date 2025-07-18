module {
  func.func @main(%arg0: tensor<13x96x74x87x45xi1>, %arg1: tensor<13x1x74x1x1xi1>) -> tensor<13x96x74x87x45xi1> {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<13x96x74x87x45xi1>, tensor<13x1x74x1x1xi1>) -> tensor<13x96x74x87x45xi1>
    %1 = tosa.identity %0 : (tensor<13x96x74x87x45xi1>) -> tensor<13x96x74x87x45xi1>
    return %1 : tensor<13x96x74x87x45xi1>
  }
}
