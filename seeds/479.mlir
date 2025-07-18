module {
  func.func @main(%arg0: tensor<95xf32>, %arg1: tensor<95xf32>, %arg2: tensor<66x83x50xi64>, %arg3: tensor<66x83x1xi64>) -> (tensor<95xf32>, tensor<66x83x50xi64>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<95xf32>, tensor<95xf32>) -> tensor<95xf32>
    %1 = tosa.sub %0, %0 : (tensor<95xf32>, tensor<95xf32>) -> tensor<95xf32>
    %2 = tosa.logical_right_shift %arg2, %arg3 : (tensor<66x83x50xi64>, tensor<66x83x1xi64>) -> tensor<66x83x50xi64>
    %3 = tosa.arithmetic_right_shift %2, %2 {round = false} : (tensor<66x83x50xi64>, tensor<66x83x50xi64>) -> tensor<66x83x50xi64>
    return %1, %3 : tensor<95xf32>, tensor<66x83x50xi64>
  }
}
