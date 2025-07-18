module {
  func.func @main(%arg0: tensor<4x79x10x34xf32>) -> tensor<4x79x10x34xf32> {
    %0 = tosa.log %arg0 : (tensor<4x79x10x34xf32>) -> tensor<4x79x10x34xf32>
    return %0 : tensor<4x79x10x34xf32>
  }
}
