module {
  func.func @main(%arg0: tensor<69x25x52x12x45xi16>, %arg1: tensor<71x43x42x98x15x42xi32>, %arg2: tensor<71x43x1x1x15x1xi32>) -> (tensor<69x25x52x12x45xi16>, tensor<71x43x42x98x15x42xi32>) {
    %0 = tosa.abs %arg0 : (tensor<69x25x52x12x45xi16>) -> tensor<69x25x52x12x45xi16>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<71x43x42x98x15x42xi32>, tensor<71x43x1x1x15x1xi32>) -> tensor<71x43x42x98x15x42xi32>
    %2 = tosa.bitwise_or %1, %1 : (tensor<71x43x42x98x15x42xi32>, tensor<71x43x42x98x15x42xi32>) -> tensor<71x43x42x98x15x42xi32>
    %3 = tosa.maximum %2, %2 : (tensor<71x43x42x98x15x42xi32>, tensor<71x43x42x98x15x42xi32>) -> tensor<71x43x42x98x15x42xi32>
    %4 = tosa.arithmetic_right_shift %3, %1 {round = false} : (tensor<71x43x42x98x15x42xi32>, tensor<71x43x42x98x15x42xi32>) -> tensor<71x43x42x98x15x42xi32>
    return %0, %4 : tensor<69x25x52x12x45xi16>, tensor<71x43x42x98x15x42xi32>
  }
}
