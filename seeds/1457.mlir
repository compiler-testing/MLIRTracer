module {
  func.func @main(%arg0: tensor<40x28x84x9x28xi64>, %arg1: tensor<1x1x1x9x1xi64>) -> tensor<40x28x84x9x28xi64> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<40x28x84x9x28xi64>, tensor<1x1x1x9x1xi64>) -> tensor<40x28x84x9x28xi64>
    %1 = tosa.clz %0 : (tensor<40x28x84x9x28xi64>) -> tensor<40x28x84x9x28xi64>
    return %1 : tensor<40x28x84x9x28xi64>
  }
}
