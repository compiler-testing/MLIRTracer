module {
  func.func @main(%arg0: tensor<39x74x39x13xi32>, %arg1: tensor<1x1x1x13xi32>) -> tensor<39x74x39x13xi32> {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<39x74x39x13xi32>, tensor<1x1x1x13xi32>) -> tensor<39x74x39x13xi32>
    return %0 : tensor<39x74x39x13xi32>
  }
}
