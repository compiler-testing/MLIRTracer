module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<37x1xi64>, %arg3: tensor<1x1xi64>, %arg4: tensor<97x65x71x97x72x74xf32>) -> (tensor<i1>, tensor<74x1xi64>, tensor<37x1xi64>, tensor<97x65x71x97x72x74xf32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.minimum %arg2, %arg3 : (tensor<37x1xi64>, tensor<1x1xi64>) -> tensor<37x1xi64>
    %3 = tosa.concat %2, %2 {axis = 0 : i32} : (tensor<37x1xi64>, tensor<37x1xi64>) -> tensor<74x1xi64>
    %4 = tosa.maximum %2, %2 : (tensor<37x1xi64>, tensor<37x1xi64>) -> tensor<37x1xi64>
    %5 = tosa.reciprocal %arg4 : (tensor<97x65x71x97x72x74xf32>) -> tensor<97x65x71x97x72x74xf32>
    return %1, %3, %4, %5 : tensor<i1>, tensor<74x1xi64>, tensor<37x1xi64>, tensor<97x65x71x97x72x74xf32>
  }
}
