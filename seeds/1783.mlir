module {
  func.func @main(%arg0: tensor<6x68x11x41x16xi8>, %arg1: tensor<6x68x11x45x16xi8>) -> tensor<6x68x11x86x16xi1> {
    %0 = tosa.concat %arg0, %arg1 {axis = 3 : i32} : (tensor<6x68x11x41x16xi8>, tensor<6x68x11x45x16xi8>) -> tensor<6x68x11x86x16xi8>
    %1 = tosa.equal %0, %0 : (tensor<6x68x11x86x16xi8>, tensor<6x68x11x86x16xi8>) -> tensor<6x68x11x86x16xi1>
    return %1 : tensor<6x68x11x86x16xi1>
  }
}
