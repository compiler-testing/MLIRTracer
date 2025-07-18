module {
  func.func @main(%arg0: tensor<5x31x42xi1>, %arg1: tensor<5x1x1xi1>) -> tensor<5x31x42xi1> {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<5x31x42xi1>, tensor<5x1x1xi1>) -> tensor<5x31x42xi1>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<5x31x42xi1>, tensor<5x31x42xi1>) -> tensor<5x31x42xi1>
    %2 = tosa.reverse %1 {axis = 2 : i32} : (tensor<5x31x42xi1>) -> tensor<5x31x42xi1>
    return %2 : tensor<5x31x42xi1>
  }
}
