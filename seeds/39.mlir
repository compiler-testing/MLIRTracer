module {
  func.func @main(%arg0: tensor<13xi16>, %arg1: tensor<82xi16>, %arg2: tensor<89xi1>, %arg3: tensor<89xi1>) -> (tensor<95xi16>, tensor<89xi1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13xi16>, tensor<82xi16>) -> tensor<95xi16>
    %1 = tosa.logical_and %arg2, %arg3 : (tensor<89xi1>, tensor<89xi1>) -> tensor<89xi1>
    return %0, %1 : tensor<95xi16>, tensor<89xi1>
  }
}
