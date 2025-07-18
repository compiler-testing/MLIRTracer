module {
  func.func @main(%arg0: tensor<50x65xi64>, %arg1: tensor<50x65xi64>) -> tensor<50x65xi1> {
    %0 = tosa.greater %arg0, %arg1 : (tensor<50x65xi64>, tensor<50x65xi64>) -> tensor<50x65xi1>
    %1 = tosa.bitwise_not %0 : (tensor<50x65xi1>) -> tensor<50x65xi1>
    return %1 : tensor<50x65xi1>
  }
}
