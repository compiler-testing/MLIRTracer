module {
  func.func @main(%arg0: tensor<93x97x15x72xi16>, %arg1: tensor<23x89x16x60x5x5xi32>, %arg2: tensor<23x1x1x60x1x5xi32>) -> (tensor<93x97x15x1xi16>, tensor<23x89x16x60x5x5xi1>) {
    %0 = tosa.reduce_product %arg0 {axis = 3 : i32} : (tensor<93x97x15x72xi16>) -> tensor<93x97x15x1xi16>
    %1 = tosa.add %0, %0 : (tensor<93x97x15x1xi16>, tensor<93x97x15x1xi16>) -> tensor<93x97x15x1xi16>
    %2 = tosa.bitwise_not %1 : (tensor<93x97x15x1xi16>) -> tensor<93x97x15x1xi16>
    %3 = tosa.minimum %arg1, %arg2 : (tensor<23x89x16x60x5x5xi32>, tensor<23x1x1x60x1x5xi32>) -> tensor<23x89x16x60x5x5xi32>
    %4 = tosa.equal %3, %3 : (tensor<23x89x16x60x5x5xi32>, tensor<23x89x16x60x5x5xi32>) -> tensor<23x89x16x60x5x5xi1>
    %5 = tosa.logical_or %4, %4 : (tensor<23x89x16x60x5x5xi1>, tensor<23x89x16x60x5x5xi1>) -> tensor<23x89x16x60x5x5xi1>
    return %2, %5 : tensor<93x97x15x1xi16>, tensor<23x89x16x60x5x5xi1>
  }
}
