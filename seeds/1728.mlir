module {
  func.func @main(%arg0: tensor<38x55x6x25xi1>, %arg1: tensor<67x79x35x32xf32>) -> (tensor<38x55x1x25xi1>, tensor<67x79x35x32xf32>) {
    %0 = tosa.reduce_any %arg0 {axis = 2 : i32} : (tensor<38x55x6x25xi1>) -> tensor<38x55x1x25xi1>
    %1 = tosa.logical_not %0 : (tensor<38x55x1x25xi1>) -> tensor<38x55x1x25xi1>
    %2 = tosa.reciprocal %arg1 : (tensor<67x79x35x32xf32>) -> tensor<67x79x35x32xf32>
    return %1, %2 : tensor<38x55x1x25xi1>, tensor<67x79x35x32xf32>
  }
}
