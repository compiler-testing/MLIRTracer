module {
  func.func @main(%arg0: tensor<75x81x94x3x80xi64>, %arg1: tensor<75x1x1x3x80xi64>) -> tensor<75x81x94x3x80xi64> {
    %0 = tosa.add %arg0, %arg1 : (tensor<75x81x94x3x80xi64>, tensor<75x1x1x3x80xi64>) -> tensor<75x81x94x3x80xi64>
    %1 = tosa.bitwise_and %0, %0 : (tensor<75x81x94x3x80xi64>, tensor<75x81x94x3x80xi64>) -> tensor<75x81x94x3x80xi64>
    return %1 : tensor<75x81x94x3x80xi64>
  }
}
