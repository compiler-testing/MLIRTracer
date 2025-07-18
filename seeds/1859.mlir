module {
  func.func @main(%arg0: tensor<41xf32>) -> tensor<41xi1> {
    %0 = tosa.abs %arg0 : (tensor<41xf32>) -> tensor<41xf32>
    %1 = tosa.abs %0 : (tensor<41xf32>) -> tensor<41xf32>
    %2 = tosa.greater %1, %1 : (tensor<41xf32>, tensor<41xf32>) -> tensor<41xi1>
    return %2 : tensor<41xi1>
  }
}
