module {
  func.func @main(%arg0: tensor<31x63x50x27xi32>, %arg1: tensor<31x63x50x27xi32>) -> tensor<31x63x50x27xi32> {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<31x63x50x27xi32>, tensor<31x63x50x27xi32>) -> tensor<31x63x50x27xi32>
    return %0 : tensor<31x63x50x27xi32>
  }
}
