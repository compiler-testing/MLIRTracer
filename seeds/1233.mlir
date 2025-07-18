module {
  func.func @main(%arg0: tensor<96xf32>) -> tensor<96xf32> {
    %0 = tosa.log %arg0 : (tensor<96xf32>) -> tensor<96xf32>
    %1 = tosa.log %0 : (tensor<96xf32>) -> tensor<96xf32>
    return %1 : tensor<96xf32>
  }
}
