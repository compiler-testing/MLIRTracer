module {
  func.func @main(%arg0: tensor<58xi32>, %arg1: tensor<1xi32>) -> tensor<58xi1> {
    %0 = tosa.equal %arg0, %arg1 : (tensor<58xi32>, tensor<1xi32>) -> tensor<58xi1>
    return %0 : tensor<58xi1>
  }
}
