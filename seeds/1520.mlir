module {
  func.func @main(%arg0: tensor<88x48x45xi16>, %arg1: tensor<5x66x19x4x78xi64>, %arg2: tensor<5x1x19x4x1xi64>, %arg3: tensor<90xi8>, %arg4: tensor<90xi8>) -> (tensor<90xi8>, tensor<88x48x45xi16>, tensor<5x132x19x4x78xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<88x48x45xi16>) -> tensor<88x48x45xi16>
    %1 = tosa.greater %arg1, %arg2 : (tensor<5x66x19x4x78xi64>, tensor<5x1x19x4x1xi64>) -> tensor<5x66x19x4x78xi1>
    %2 = tosa.minimum %arg3, %arg4 : (tensor<90xi8>, tensor<90xi8>) -> tensor<90xi8>
    %3 = tosa.bitwise_xor %1, %1 : (tensor<5x66x19x4x78xi1>, tensor<5x66x19x4x78xi1>) -> tensor<5x66x19x4x78xi1>
    %4 = tosa.concat %3, %3 {axis = 1 : i32} : (tensor<5x66x19x4x78xi1>, tensor<5x66x19x4x78xi1>) -> tensor<5x132x19x4x78xi1>
    %5 = tosa.abs %0 : (tensor<88x48x45xi16>) -> tensor<88x48x45xi16>
    %6 = tosa.sub %4, %4 : (tensor<5x132x19x4x78xi1>, tensor<5x132x19x4x78xi1>) -> tensor<5x132x19x4x78xi1>
    return %2, %5, %6 : tensor<90xi8>, tensor<88x48x45xi16>, tensor<5x132x19x4x78xi1>
  }
}
