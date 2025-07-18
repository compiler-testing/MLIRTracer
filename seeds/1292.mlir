module {
  func.func @main(%arg0: tensor<82x11x94xi64>, %arg1: tensor<22xi1>) -> (tensor<82x11x1xi64>, tensor<1xi1>) {
    %0 = tosa.reduce_min %arg0 {axis = 2 : i32} : (tensor<82x11x94xi64>) -> tensor<82x11x1xi64>
    %1 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<22xi1>) -> tensor<1xi1>
    return %0, %1 : tensor<82x11x1xi64>, tensor<1xi1>
  }
}
