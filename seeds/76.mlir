module {
  func.func @main(%arg0: tensor<92x25x92x65xi1>) -> tensor<92x25x92x65xi1> {
    %0 = tosa.logical_not %arg0 : (tensor<92x25x92x65xi1>) -> tensor<92x25x92x65xi1>
    return %0 : tensor<92x25x92x65xi1>
  }
}
