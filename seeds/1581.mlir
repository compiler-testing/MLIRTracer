module {
  func.func @main(%arg0: tensor<68x42x47x20x30xi1>, %arg1: tensor<1x1x1x20x1xi1>, %arg2: tensor<f32>, %arg3: tensor<94x11xi32>, %arg4: tensor<1x1xi32>) -> (tensor<f32>, tensor<94x11xi32>, tensor<68x42x47x20x30xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<68x42x47x20x30xi1>, tensor<1x1x1x20x1xi1>) -> tensor<68x42x47x20x30xi1>
    %1 = tosa.reciprocal %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.bitwise_xor %0, %0 : (tensor<68x42x47x20x30xi1>, tensor<68x42x47x20x30xi1>) -> tensor<68x42x47x20x30xi1>
    %3 = tosa.pow %1, %1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %4 = tosa.minimum %arg3, %arg4 : (tensor<94x11xi32>, tensor<1x1xi32>) -> tensor<94x11xi32>
    %5 = tosa.logical_or %2, %0 : (tensor<68x42x47x20x30xi1>, tensor<68x42x47x20x30xi1>) -> tensor<68x42x47x20x30xi1>
    return %3, %4, %5 : tensor<f32>, tensor<94x11xi32>, tensor<68x42x47x20x30xi1>
  }
}
