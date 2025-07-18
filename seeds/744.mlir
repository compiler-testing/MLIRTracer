module {
  func.func @main(%arg0: tensor<5x12xi16>, %arg1: tensor<1x12xi16>, %arg2: tensor<44x71x88xi1>, %arg3: tensor<1x1x1xi1>) -> (tensor<5x12xi16>, tensor<44x71x88xi1>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<5x12xi16>, tensor<1x12xi16>) -> tensor<5x12xi16>
    %1 = tosa.abs %0 : (tensor<5x12xi16>) -> tensor<5x12xi16>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<44x71x88xi1>, tensor<1x1x1xi1>) -> tensor<44x71x88xi1>
    return %1, %2 : tensor<5x12xi16>, tensor<44x71x88xi1>
  }
}
