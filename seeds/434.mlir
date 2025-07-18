module {
  func.func @main(%arg0: tensor<14x1x88x15x13xf32>, %arg1: tensor<36xi16>, %arg2: tensor<36xi16>, %arg3: tensor<82x98x43xi1>, %arg4: tensor<82x98x1xi1>) -> (tensor<14x1x88x15x13xf32>, tensor<36xi16>, tensor<82x98x43xi1>) {
    %0 = tosa.floor %arg0 : (tensor<14x1x88x15x13xf32>) -> tensor<14x1x88x15x13xf32>
    %1 = tosa.minimum %0, %0 : (tensor<14x1x88x15x13xf32>, tensor<14x1x88x15x13xf32>) -> tensor<14x1x88x15x13xf32>
    %2 = tosa.sub %1, %1 : (tensor<14x1x88x15x13xf32>, tensor<14x1x88x15x13xf32>) -> tensor<14x1x88x15x13xf32>
    %3 = tosa.logical_right_shift %arg1, %arg2 : (tensor<36xi16>, tensor<36xi16>) -> tensor<36xi16>
    %4 = tosa.bitwise_or %3, %3 : (tensor<36xi16>, tensor<36xi16>) -> tensor<36xi16>
    %5 = tosa.logical_or %arg3, %arg4 : (tensor<82x98x43xi1>, tensor<82x98x1xi1>) -> tensor<82x98x43xi1>
    return %2, %4, %5 : tensor<14x1x88x15x13xf32>, tensor<36xi16>, tensor<82x98x43xi1>
  }
}
