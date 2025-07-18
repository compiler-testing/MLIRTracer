module {
  func.func @main(%arg0: tensor<97xi1>) -> tensor<1xi1> {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<97xi1>) -> tensor<1xi1>
    %1 = tosa.logical_and %0, %0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %2 : tensor<1xi1>
  }
}
