module {
  func.func @main(%arg0: tensor<6x77x56x89x77x84xi1>, %arg1: tensor<96xf32>) -> (tensor<6x77x56x89x77x84xi1>, tensor<96xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<6x77x56x89x77x84xi1>) -> tensor<6x77x56x89x77x84xi1>
    %1 = tosa.clz %0 : (tensor<6x77x56x89x77x84xi1>) -> tensor<6x77x56x89x77x84xi1>
    %2 = tosa.exp %arg1 : (tensor<96xf32>) -> tensor<96xf32>
    %3 = tosa.rsqrt %2 : (tensor<96xf32>) -> tensor<96xf32>
    return %1, %3 : tensor<6x77x56x89x77x84xi1>, tensor<96xf32>
  }
}
