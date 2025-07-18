module {
  func.func @main(%arg0: tensor<71x15x37x59x33x46xi1>, %arg1: tensor<22x60x99xi8>) -> (tensor<71x30x37x59x33x46xi1>, tensor<22x120x99xi8>) {
    %0 = tosa.logical_not %arg0 : (tensor<71x15x37x59x33x46xi1>) -> tensor<71x15x37x59x33x46xi1>
    %1 = tosa.concat %0, %0 {axis = 1 : i32} : (tensor<71x15x37x59x33x46xi1>, tensor<71x15x37x59x33x46xi1>) -> tensor<71x30x37x59x33x46xi1>
    %2 = tosa.reverse %arg1 {axis = 2 : i32} : (tensor<22x60x99xi8>) -> tensor<22x60x99xi8>
    %3 = tosa.concat %2, %2 {axis = 1 : i32} : (tensor<22x60x99xi8>, tensor<22x60x99xi8>) -> tensor<22x120x99xi8>
    %4 = tosa.bitwise_or %3, %3 : (tensor<22x120x99xi8>, tensor<22x120x99xi8>) -> tensor<22x120x99xi8>
    return %1, %4 : tensor<71x30x37x59x33x46xi1>, tensor<22x120x99xi8>
  }
}
