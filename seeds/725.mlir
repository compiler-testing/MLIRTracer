module {
  func.func @main(%arg0: tensor<68x26xi64>, %arg1: tensor<98xf32>, %arg2: tensor<35x51x27x36xi1>) -> (tensor<98xf32>, tensor<35x51x27x36xi1>, tensor<68x26xi64>) {
    %0 = tosa.reverse %arg0 {axis = 1 : i32} : (tensor<68x26xi64>) -> tensor<68x26xi64>
    %1 = tosa.ceil %arg1 : (tensor<98xf32>) -> tensor<98xf32>
    %2 = tosa.logical_not %arg2 : (tensor<35x51x27x36xi1>) -> tensor<35x51x27x36xi1>
    %3 = tosa.bitwise_not %0 : (tensor<68x26xi64>) -> tensor<68x26xi64>
    return %1, %2, %3 : tensor<98xf32>, tensor<35x51x27x36xi1>, tensor<68x26xi64>
  }
}
