module {
  func.func @main(%arg0: tensor<46x72x34xi1>, %arg1: tensor<18x7x31x81x69xi32>, %arg2: tensor<18x1x1x81x1xi32>) -> (tensor<46x72x1xi1>, tensor<18x7x31x81x69xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 2 : i32} : (tensor<46x72x34xi1>) -> tensor<46x72x1xi1>
    %1 = tosa.greater_equal %arg1, %arg2 : (tensor<18x7x31x81x69xi32>, tensor<18x1x1x81x1xi32>) -> tensor<18x7x31x81x69xi1>
    return %0, %1 : tensor<46x72x1xi1>, tensor<18x7x31x81x69xi1>
  }
}
