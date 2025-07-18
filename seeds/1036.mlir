module {
  func.func @main(%arg0: tensor<34x53x23xi1>, %arg1: tensor<1x1x1xi1>) -> tensor<1x1x23xi1> {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<34x53x23xi1>, tensor<1x1x1xi1>) -> tensor<34x53x23xi1>
    %1 = tosa.logical_or %0, %0 : (tensor<34x53x23xi1>, tensor<34x53x23xi1>) -> tensor<34x53x23xi1>
    %2 = tosa.reduce_all %1 {axis = 1 : i32} : (tensor<34x53x23xi1>) -> tensor<34x1x23xi1>
    %3 = tosa.logical_right_shift %2, %2 : (tensor<34x1x23xi1>, tensor<34x1x23xi1>) -> tensor<34x1x23xi1>
    %4 = tosa.reduce_min %3 {axis = 0 : i32} : (tensor<34x1x23xi1>) -> tensor<1x1x23xi1>
    %5 = tosa.bitwise_or %4, %4 : (tensor<1x1x23xi1>, tensor<1x1x23xi1>) -> tensor<1x1x23xi1>
    return %5 : tensor<1x1x23xi1>
  }
}
