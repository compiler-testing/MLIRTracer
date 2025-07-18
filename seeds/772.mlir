module {
  func.func @main(%arg0: tensor<89x81xi64>, %arg1: tensor<1x81xi64>, %arg2: tensor<87x88x36x72x66xf32>) -> (tensor<89x81xi64>, tensor<87x88x36x72x66xf32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<89x81xi64>, tensor<1x81xi64>) -> tensor<89x81xi64>
    %1 = tosa.clz %0 : (tensor<89x81xi64>) -> tensor<89x81xi64>
    %2 = tosa.reciprocal %arg2 : (tensor<87x88x36x72x66xf32>) -> tensor<87x88x36x72x66xf32>
    return %1, %2 : tensor<89x81xi64>, tensor<87x88x36x72x66xf32>
  }
}
