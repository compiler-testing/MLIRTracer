module {
  func.func @main(%arg0: tensor<75x16xi1>, %arg1: tensor<52x50x90x65x5x89xf32>) -> (tensor<150x1xi1>, tensor<52x50x90x65x5x89xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<75x16xi1>) -> tensor<75x16xi1>
    %1 = tosa.logical_or %0, %0 : (tensor<75x16xi1>, tensor<75x16xi1>) -> tensor<75x16xi1>
    %2 = tosa.reduce_all %1 {axis = 1 : i32} : (tensor<75x16xi1>) -> tensor<75x1xi1>
    %3 = tosa.floor %arg1 : (tensor<52x50x90x65x5x89xf32>) -> tensor<52x50x90x65x5x89xf32>
    %4 = tosa.concat %2, %2 {axis = 0 : i32} : (tensor<75x1xi1>, tensor<75x1xi1>) -> tensor<150x1xi1>
    %5 = tosa.identity %4 : (tensor<150x1xi1>) -> tensor<150x1xi1>
    %6 = tosa.ceil %3 : (tensor<52x50x90x65x5x89xf32>) -> tensor<52x50x90x65x5x89xf32>
    return %5, %6 : tensor<150x1xi1>, tensor<52x50x90x65x5x89xf32>
  }
}
