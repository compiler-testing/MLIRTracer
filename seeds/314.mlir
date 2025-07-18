module {
  func.func @main(%arg0: tensor<33x56x5x93x94xi16>, %arg1: tensor<95x97x32xi1>, %arg2: tensor<1x1x1xi1>) -> (tensor<33x56x5x93x94xi16>, tensor<95x97x32xi1>) {
    %0 = tosa.identity %arg0 : (tensor<33x56x5x93x94xi16>) -> tensor<33x56x5x93x94xi16>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<95x97x32xi1>, tensor<1x1x1xi1>) -> tensor<95x97x32xi1>
    return %0, %1 : tensor<33x56x5x93x94xi16>, tensor<95x97x32xi1>
  }
}
