module {
  func.func @main(%arg0: tensor<68x67x87xi64>, %arg1: tensor<1x67x1xi64>) -> tensor<67x87xi32> {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<68x67x87xi64>, tensor<1x67x1xi64>) -> tensor<68x67x87xi64>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<68x67x87xi64>) -> tensor<67x87xi32>
    return %1 : tensor<67x87xi32>
  }
}
