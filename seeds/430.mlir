module {
  func.func @main(%arg0: tensor<42x47x68x59x28x9xi64>, %arg1: tensor<1x1x68x1x28x9xi64>) -> tensor<42x47x68x59x56x9xi64> {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<42x47x68x59x28x9xi64>, tensor<1x1x68x1x28x9xi64>) -> tensor<42x47x68x59x28x9xi64>
    %1 = tosa.concat %0, %0 {axis = 4 : i32} : (tensor<42x47x68x59x28x9xi64>, tensor<42x47x68x59x28x9xi64>) -> tensor<42x47x68x59x56x9xi64>
    return %1 : tensor<42x47x68x59x56x9xi64>
  }
}
