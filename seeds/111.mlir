module {
  func.func @main(%arg0: tensor<81x56xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<50x48xi1>) -> (tensor<50x1xi1>, tensor<81x56xf32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<81x56xf32>, tensor<1x1xf32>) -> tensor<81x56xf32>
    %1 = tosa.reduce_any %arg2 {axis = 1 : i32} : (tensor<50x48xi1>) -> tensor<50x1xi1>
    %2 = tosa.log %0 : (tensor<81x56xf32>) -> tensor<81x56xf32>
    return %1, %2 : tensor<50x1xi1>, tensor<81x56xf32>
  }
}
