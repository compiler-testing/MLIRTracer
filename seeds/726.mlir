module {
  func.func @main(%arg0: tensor<23x1x87xi1>, %arg1: tensor<1x1x1xi1>) -> tensor<23x1x1xi1> {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<23x1x87xi1>, tensor<1x1x1xi1>) -> tensor<23x1x87xi1>
    %1 = tosa.reduce_sum %0 {axis = 2 : i32} : (tensor<23x1x87xi1>) -> tensor<23x1x1xi1>
    return %1 : tensor<23x1x1xi1>
  }
}
