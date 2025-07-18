module {
  func.func @main(%arg0: tensor<72x38x51x90xi1>, %arg1: tensor<72x38x1x90xi1>, %arg2: tensor<94x27xf32>) -> (tensor<94x27xf32>, tensor<72x38x51x90xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<72x38x51x90xi1>, tensor<72x38x1x90xi1>) -> tensor<72x38x51x90xi1>
    %1 = tosa.reciprocal %arg2 : (tensor<94x27xf32>) -> tensor<94x27xf32>
    %2 = tosa.identity %0 : (tensor<72x38x51x90xi1>) -> tensor<72x38x51x90xi1>
    return %1, %2 : tensor<94x27xf32>, tensor<72x38x51x90xi1>
  }
}
