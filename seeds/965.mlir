module {
  func.func @main(%arg0: tensor<100xi8>, %arg1: tensor<44x53x79x70xi1>, %arg2: tensor<44x1x79x70xi1>) -> (tensor<100xi8>, tensor<44x53x79x70xi1>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<100xi8>) -> tensor<100xi8>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<44x53x79x70xi1>, tensor<44x1x79x70xi1>) -> tensor<44x53x79x70xi1>
    return %0, %1 : tensor<100xi8>, tensor<44x53x79x70xi1>
  }
}
