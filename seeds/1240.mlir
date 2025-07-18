module {
  func.func @main(%arg0: tensor<4x28x67xi8>, %arg1: tensor<4x1x67xi8>, %arg2: tensor<63x24x40xf32>) -> (tensor<4x28x67xi8>, tensor<63x24x40xf32>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<4x28x67xi8>, tensor<4x1x67xi8>) -> tensor<4x28x67xi8>
    %1 = tosa.ceil %arg2 : (tensor<63x24x40xf32>) -> tensor<63x24x40xf32>
    return %0, %1 : tensor<4x28x67xi8>, tensor<63x24x40xf32>
  }
}
