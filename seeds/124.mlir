module {
  func.func @main(%arg0: tensor<28x19x100x44x41x47xi1>, %arg1: tensor<1x19x100x44x1x47xi1>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<28x19x100x44x41x47xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<28x19x100x44x41x47xi1>, tensor<1x19x100x44x1x47xi1>) -> tensor<28x19x100x44x41x47xi1>
    %1 = tosa.logical_xor %0, %0 : (tensor<28x19x100x44x41x47xi1>, tensor<28x19x100x44x41x47xi1>) -> tensor<28x19x100x44x41x47xi1>
    %2 = tosa.pow %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = tosa.bitwise_and %1, %1 : (tensor<28x19x100x44x41x47xi1>, tensor<28x19x100x44x41x47xi1>) -> tensor<28x19x100x44x41x47xi1>
    return %2, %3 : tensor<f32>, tensor<28x19x100x44x41x47xi1>
  }
}
