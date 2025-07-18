module {
  func.func @main(%arg0: tensor<6xf32>, %arg1: tensor<29xi32>, %arg2: tensor<29xi32>) -> (tensor<29xi32>, tensor<6xf32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<6xf32>) -> tensor<6xf32>
    %1 = tosa.identity %0 : (tensor<6xf32>) -> tensor<6xf32>
    %2 = tosa.bitwise_and %arg1, %arg2 : (tensor<29xi32>, tensor<29xi32>) -> tensor<29xi32>
    %3 = tosa.minimum %1, %0 : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
    return %2, %3 : tensor<29xi32>, tensor<6xf32>
  }
}
