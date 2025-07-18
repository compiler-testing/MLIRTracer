module {
  func.func @main(%arg0: tensor<43x88x53x38x80xi64>, %arg1: tensor<43x88x1x1x1xi64>, %arg2: tensor<10xi1>) -> (tensor<43x88x53x38x80xi64>, tensor<1xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<43x88x53x38x80xi64>, tensor<43x88x1x1x1xi64>) -> tensor<43x88x53x38x80xi64>
    %1 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<10xi1>) -> tensor<1xi1>
    return %0, %1 : tensor<43x88x53x38x80xi64>, tensor<1xi1>
  }
}
