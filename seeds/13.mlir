module {
  func.func @main(%arg0: tensor<61xf32>) -> tensor<61xf32> {
    %0 = tosa.tanh %arg0 : (tensor<61xf32>) -> tensor<61xf32>
    %1 = tosa.add %0, %0 : (tensor<61xf32>, tensor<61xf32>) -> tensor<61xf32>
    %2 = tosa.tanh %1 : (tensor<61xf32>) -> tensor<61xf32>
    %3 = tosa.exp %2 : (tensor<61xf32>) -> tensor<61xf32>
    return %3 : tensor<61xf32>
  }
}
