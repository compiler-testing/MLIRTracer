module {
  func.func @main(%arg0: tensor<90xi1>, %arg1: tensor<53x68xf32>) -> (tensor<90xi1>, tensor<53x68xf32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<90xi1>) -> tensor<90xi1>
    %1 = tosa.ceil %arg1 : (tensor<53x68xf32>) -> tensor<53x68xf32>
    return %0, %1 : tensor<90xi1>, tensor<53x68xf32>
  }
}
