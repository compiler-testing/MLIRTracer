module {
  func.func @main(%arg0: tensor<42x97xi64>) -> tensor<42x97xi1> {
    %0 = tosa.reverse %arg0 {axis = 1 : i32} : (tensor<42x97xi64>) -> tensor<42x97xi64>
    %1 = tosa.greater %0, %0 : (tensor<42x97xi64>, tensor<42x97xi64>) -> tensor<42x97xi1>
    %2 = tosa.logical_and %1, %1 : (tensor<42x97xi1>, tensor<42x97xi1>) -> tensor<42x97xi1>
    return %2 : tensor<42x97xi1>
  }
}
