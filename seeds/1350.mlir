module {
  func.func @main(%arg0: tensor<73x30xi64>, %arg1: tensor<73x1xi64>, %arg2: tensor<16x21x67x89x33xi1>, %arg3: tensor<16x1x1x1x33xi1>, %arg4: tensor<f32>) -> (tensor<16x21x67x89x33xi1>, tensor<73x30xi64>, tensor<f32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<73x30xi64>, tensor<73x1xi64>) -> tensor<73x30xi64>
    %1 = tosa.logical_or %arg2, %arg3 : (tensor<16x21x67x89x33xi1>, tensor<16x1x1x1x33xi1>) -> tensor<16x21x67x89x33xi1>
    %2 = tosa.bitwise_not %0 : (tensor<73x30xi64>) -> tensor<73x30xi64>
    %3 = tosa.tanh %arg4 : (tensor<f32>) -> tensor<f32>
    return %1, %2, %3 : tensor<16x21x67x89x33xi1>, tensor<73x30xi64>, tensor<f32>
  }
}
