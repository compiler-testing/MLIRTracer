module {
  func.func @main(%arg0: tensor<83xi32>, %arg1: tensor<1xi32>) -> tensor<83xi32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<83xi32>, tensor<1xi32>) -> tensor<83xi32>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<83xi32>, tensor<83xi32>) -> tensor<83xi32>
    return %1 : tensor<83xi32>
  }
}
