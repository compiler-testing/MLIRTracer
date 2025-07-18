module {
  func.func @main(%arg0: tensor<59x4x5x6x3x39xi64>, %arg1: tensor<59x4x5x6x1x39xi64>, %arg2: tensor<81x77xi64>, %arg3: tensor<42x23x31x18xf32>) -> (tensor<59x4x5x6x3x39xi64>, tensor<77xi32>, tensor<42x23x1x18xf32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<59x4x5x6x3x39xi64>, tensor<59x4x5x6x1x39xi64>) -> tensor<59x4x5x6x3x39xi64>
    %1 = tosa.argmax %arg2 {axis = 0 : i32} : (tensor<81x77xi64>) -> tensor<77xi32>
    %2 = tosa.tanh %arg3 : (tensor<42x23x31x18xf32>) -> tensor<42x23x31x18xf32>
    %3 = tosa.reduce_max %2 {axis = 2 : i32} : (tensor<42x23x31x18xf32>) -> tensor<42x23x1x18xf32>
    return %0, %1, %3 : tensor<59x4x5x6x3x39xi64>, tensor<77xi32>, tensor<42x23x1x18xf32>
  }
}
