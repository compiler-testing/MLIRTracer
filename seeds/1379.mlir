module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
    %0 = tosa.equal %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %1 = tosa.bitwise_not %0 : (tensor<i1>) -> tensor<i1>
    return %1 : tensor<i1>
  }
}
