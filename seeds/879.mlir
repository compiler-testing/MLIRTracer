module {
  func.func @main(%arg0: tensor<67x81xi64>, %arg1: tensor<5x73x46x66xi1>, %arg2: tensor<1x73x1x1xi1>) -> (tensor<1x81xi64>, tensor<5x73x46x66xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<67x81xi64>) -> tensor<1x81xi64>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<5x73x46x66xi1>, tensor<1x73x1x1xi1>) -> tensor<5x73x46x66xi1>
    return %0, %1 : tensor<1x81xi64>, tensor<5x73x46x66xi1>
  }
}
