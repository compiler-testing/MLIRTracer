module {
  func.func @main(%arg0: tensor<97x58xi64>, %arg1: tensor<76x58xi64>) -> tensor<173x58xi64> {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<97x58xi64>, tensor<76x58xi64>) -> tensor<173x58xi64>
    %1 = tosa.maximum %0, %0 : (tensor<173x58xi64>, tensor<173x58xi64>) -> tensor<173x58xi64>
    return %1 : tensor<173x58xi64>
  }
}
