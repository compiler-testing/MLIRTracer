module {
  func.func @main(%arg0: tensor<48x51x46xi32>) -> tensor<48x51x46xi32> {
    %0 = tosa.abs %arg0 : (tensor<48x51x46xi32>) -> tensor<48x51x46xi32>
    return %0 : tensor<48x51x46xi32>
  }
}
