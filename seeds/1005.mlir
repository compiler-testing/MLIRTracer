module {
  func.func @main(%arg0: tensor<61x25x45x39x90xi32>, %arg1: tensor<1x25x1x39x90xi32>) -> tensor<61x25x45x39x90xi32> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<61x25x45x39x90xi32>, tensor<1x25x1x39x90xi32>) -> tensor<61x25x45x39x90xi32>
    %1 = tosa.bitwise_and %0, %0 : (tensor<61x25x45x39x90xi32>, tensor<61x25x45x39x90xi32>) -> tensor<61x25x45x39x90xi32>
    return %1 : tensor<61x25x45x39x90xi32>
  }
}
