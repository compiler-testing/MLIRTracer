module {
  func.func @main(%arg0: tensor<1x93x57x73xi1>, %arg1: tensor<92x3xi8>, %arg2: tensor<1x3xi8>) -> (tensor<1x1x57x73xi1>, tensor<92x3xi1>) {
    %0 = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<1x93x57x73xi1>) -> tensor<1x1x57x73xi1>
    %1 = tosa.greater %arg1, %arg2 : (tensor<92x3xi8>, tensor<1x3xi8>) -> tensor<92x3xi1>
    return %0, %1 : tensor<1x1x57x73xi1>, tensor<92x3xi1>
  }
}
