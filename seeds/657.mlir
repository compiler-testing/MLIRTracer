module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<100x31x68x36xf32>) -> (tensor<100x31x68x36xf32>, tensor<i1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.log %arg2 : (tensor<100x31x68x36xf32>) -> tensor<100x31x68x36xf32>
    %2 = tosa.logical_and %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.logical_or %2, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.logical_or %3, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.logical_or %4, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %1, %5 : tensor<100x31x68x36xf32>, tensor<i1>
  }
}
