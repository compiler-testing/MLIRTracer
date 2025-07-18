module {
  func.func @main(%arg0: tensor<77x31x93x71x26xf32>, %arg1: tensor<77x11x93x71x26xf32>, %arg2: tensor<85x72xi64>) -> (tensor<77x42x93x71x26xf32>, tensor<85x1xi64>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<77x31x93x71x26xf32>, tensor<77x11x93x71x26xf32>) -> tensor<77x42x93x71x26xf32>
    %1 = tosa.clz %arg2 : (tensor<85x72xi64>) -> tensor<85x72xi64>
    %2 = tosa.maximum %1, %1 : (tensor<85x72xi64>, tensor<85x72xi64>) -> tensor<85x72xi64>
    %3 = tosa.reduce_min %2 {axis = 1 : i32} : (tensor<85x72xi64>) -> tensor<85x1xi64>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<85x1xi64>, tensor<85x1xi64>) -> tensor<85x1xi64>
    %5 = tosa.minimum %4, %4 : (tensor<85x1xi64>, tensor<85x1xi64>) -> tensor<85x1xi64>
    return %0, %5 : tensor<77x42x93x71x26xf32>, tensor<85x1xi64>
  }
}
