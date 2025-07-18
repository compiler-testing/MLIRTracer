module {
  func.func @main(%arg0: tensor<38x31x71x6x27x46xf32>, %arg1: tensor<82xi64>) -> (tensor<38x31x71x6x27x46xi1>, tensor<38x31x71x6x27x46xf32>, tensor<1xi64>) {
    %0 = tosa.sigmoid %arg0 : (tensor<38x31x71x6x27x46xf32>) -> tensor<38x31x71x6x27x46xf32>
    %1 = tosa.minimum %0, %0 : (tensor<38x31x71x6x27x46xf32>, tensor<38x31x71x6x27x46xf32>) -> tensor<38x31x71x6x27x46xf32>
    %2 = tosa.greater_equal %1, %0 : (tensor<38x31x71x6x27x46xf32>, tensor<38x31x71x6x27x46xf32>) -> tensor<38x31x71x6x27x46xi1>
    %3 = tosa.maximum %0, %1 : (tensor<38x31x71x6x27x46xf32>, tensor<38x31x71x6x27x46xf32>) -> tensor<38x31x71x6x27x46xf32>
    %4 = tosa.reduce_product %arg1 {axis = 0 : i32} : (tensor<82xi64>) -> tensor<1xi64>
    return %2, %3, %4 : tensor<38x31x71x6x27x46xi1>, tensor<38x31x71x6x27x46xf32>, tensor<1xi64>
  }
}
