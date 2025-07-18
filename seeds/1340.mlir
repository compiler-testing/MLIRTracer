module {
  func.func @main(%arg0: tensor<27x87xi1>, %arg1: tensor<1x87xi1>) -> tensor<1x1xi1> {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<27x87xi1>, tensor<1x87xi1>) -> tensor<27x87xi1>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<27x87xi1>) -> tensor<27x1xi1>
    %2 = tosa.reduce_all %1 {axis = 0 : i32} : (tensor<27x1xi1>) -> tensor<1x1xi1>
    %3 = tosa.bitwise_not %2 : (tensor<1x1xi1>) -> tensor<1x1xi1>
    return %3 : tensor<1x1xi1>
  }
}
