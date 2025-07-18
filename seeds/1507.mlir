module {
  func.func @main(%arg0: tensor<98x35xf32>, %arg1: tensor<98x35xf32>) -> tensor<98x1xi1> {
    %0 = tosa.pow %arg0, %arg1 : (tensor<98x35xf32>, tensor<98x35xf32>) -> tensor<98x35xf32>
    %1 = tosa.greater %0, %0 : (tensor<98x35xf32>, tensor<98x35xf32>) -> tensor<98x35xi1>
    %2 = tosa.reduce_all %1 {axis = 1 : i32} : (tensor<98x35xi1>) -> tensor<98x1xi1>
    return %2 : tensor<98x1xi1>
  }
}
