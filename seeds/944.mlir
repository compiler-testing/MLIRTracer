module {
  func.func @main(%arg0: tensor<76x72x98x37x65x85xi32>, %arg1: tensor<76x72x1x37x65x1xi32>) -> tensor<76x72x98x37x65x85xi32> {
    %0 = tosa.sub %arg0, %arg1 : (tensor<76x72x98x37x65x85xi32>, tensor<76x72x1x37x65x1xi32>) -> tensor<76x72x98x37x65x85xi32>
    return %0 : tensor<76x72x98x37x65x85xi32>
  }
}
