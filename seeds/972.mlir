module {
  func.func @main(%arg0: tensor<45x2x66x59x51x43xi64>, %arg1: tensor<45x2x66x59x51x53xi64>, %arg2: tensor<i1>, %arg3: tensor<i1>) -> (tensor<45x2x66x59x51x96xi64>, tensor<i1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 5 : i32} : (tensor<45x2x66x59x51x43xi64>, tensor<45x2x66x59x51x53xi64>) -> tensor<45x2x66x59x51x96xi64>
    %1 = tosa.logical_xor %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.logical_or %1, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %0, %2 : tensor<45x2x66x59x51x96xi64>, tensor<i1>
  }
}
