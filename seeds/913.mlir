module {
  func.func @main(%arg0: tensor<98x7xf32>) -> tensor<98x7xi1> {
    %0 = tosa.abs %arg0 : (tensor<98x7xf32>) -> tensor<98x7xf32>
    %1 = tosa.equal %0, %0 : (tensor<98x7xf32>, tensor<98x7xf32>) -> tensor<98x7xi1>
    return %1 : tensor<98x7xi1>
  }
}
