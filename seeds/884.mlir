module {
  func.func @main(%arg0: tensor<36x30xi32>, %arg1: tensor<19xi1>, %arg2: tensor<1xi1>) -> (tensor<36x30xi32>, tensor<1xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<36x30xi32>) -> tensor<36x30xi32>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<19xi1>, tensor<1xi1>) -> tensor<19xi1>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<19xi1>) -> tensor<1xi1>
    return %0, %2 : tensor<36x30xi32>, tensor<1xi1>
  }
}
