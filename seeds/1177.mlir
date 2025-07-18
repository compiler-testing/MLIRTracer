module {
  func.func @main(%arg0: tensor<19x57x78x27xf32>, %arg1: tensor<39x23xi32>, %arg2: tensor<39x23xi32>) -> (tensor<39x23xi32>, tensor<19x57x78x27xf32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<19x57x78x27xf32>) -> tensor<19x57x78x27xf32>
    %1 = tosa.bitwise_and %arg1, %arg2 : (tensor<39x23xi32>, tensor<39x23xi32>) -> tensor<39x23xi32>
    %2 = tosa.pow %0, %0 : (tensor<19x57x78x27xf32>, tensor<19x57x78x27xf32>) -> tensor<19x57x78x27xf32>
    return %1, %2 : tensor<39x23xi32>, tensor<19x57x78x27xf32>
  }
}
