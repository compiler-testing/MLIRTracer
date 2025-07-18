module {
  func.func @main(%arg0: tensor<7xf32>, %arg1: tensor<46xi1>, %arg2: tensor<46xi1>) -> (tensor<7xf32>, tensor<46xi1>) {
    %0 = tosa.log %arg0 : (tensor<7xf32>) -> tensor<7xf32>
    %1 = tosa.ceil %0 : (tensor<7xf32>) -> tensor<7xf32>
    %2 = tosa.logical_and %arg1, %arg2 : (tensor<46xi1>, tensor<46xi1>) -> tensor<46xi1>
    %3 = tosa.bitwise_and %2, %2 : (tensor<46xi1>, tensor<46xi1>) -> tensor<46xi1>
    return %1, %3 : tensor<7xf32>, tensor<46xi1>
  }
}
