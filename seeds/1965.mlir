module {
  func.func @main(%arg0: tensor<96x84xi32>, %arg1: tensor<96x1xi32>, %arg2: tensor<66xi1>, %arg3: tensor<66xi1>) -> (tensor<96x84xi32>, tensor<1xi1>, tensor<66xi1>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<96x84xi32>, tensor<96x1xi32>) -> tensor<96x84xi32>
    %1 = tosa.logical_xor %arg2, %arg3 : (tensor<66xi1>, tensor<66xi1>) -> tensor<66xi1>
    %2 = tosa.reduce_all %1 {axis = 0 : i32} : (tensor<66xi1>) -> tensor<1xi1>
    %3 = tosa.sub %1, %1 : (tensor<66xi1>, tensor<66xi1>) -> tensor<66xi1>
    return %0, %2, %3 : tensor<96x84xi32>, tensor<1xi1>, tensor<66xi1>
  }
}
