module {
  func.func @main(%arg0: tensor<37xi32>, %arg1: tensor<37xi32>, %arg2: tensor<33xf32>, %arg3: tensor<1xf32>) -> (tensor<37xi1>, tensor<33xf32>, tensor<33xf32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<37xi32>, tensor<37xi32>) -> tensor<37xi32>
    %1 = tosa.greater %0, %0 : (tensor<37xi32>, tensor<37xi32>) -> tensor<37xi1>
    %2 = tosa.logical_or %1, %1 : (tensor<37xi1>, tensor<37xi1>) -> tensor<37xi1>
    %3 = tosa.pow %arg2, %arg3 : (tensor<33xf32>, tensor<1xf32>) -> tensor<33xf32>
    %4 = tosa.tanh %3 : (tensor<33xf32>) -> tensor<33xf32>
    %5 = tosa.reverse %3 {axis = 0 : i32} : (tensor<33xf32>) -> tensor<33xf32>
    return %2, %4, %5 : tensor<37xi1>, tensor<33xf32>, tensor<33xf32>
  }
}
