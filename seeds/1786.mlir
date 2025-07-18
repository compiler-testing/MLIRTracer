module {
  func.func @main(%arg0: tensor<7xf32>, %arg1: tensor<7xf32>) -> tensor<7xf32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<7xf32>, tensor<7xf32>) -> tensor<7xf32>
    return %0 : tensor<7xf32>
  }
}
