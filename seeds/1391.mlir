module {
  func.func @main(%arg0: tensor<81x24x12x81x65xi1>, %arg1: tensor<81x1x12x81x1xi1>) -> tensor<81x24x12x81x65xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<81x24x12x81x65xi1>, tensor<81x1x12x81x1xi1>) -> tensor<81x24x12x81x65xi1>
    return %0 : tensor<81x24x12x81x65xi1>
  }
}
