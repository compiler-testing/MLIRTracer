module {
  func.func @main(%arg0: tensor<20xi1>, %arg1: tensor<20xi1>) -> tensor<20xi1> {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<20xi1>, tensor<20xi1>) -> tensor<20xi1>
    %1 = tosa.logical_not %0 : (tensor<20xi1>) -> tensor<20xi1>
    return %1 : tensor<20xi1>
  }
}
