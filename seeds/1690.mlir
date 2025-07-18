module {
  func.func @main(%arg0: tensor<7x1xi1>, %arg1: tensor<17x1xi1>) -> tensor<48x1xi1> {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<7x1xi1>, tensor<17x1xi1>) -> tensor<24x1xi1>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<24x1xi1>, tensor<24x1xi1>) -> tensor<48x1xi1>
    return %1 : tensor<48x1xi1>
  }
}
