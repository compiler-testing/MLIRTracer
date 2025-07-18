module {
  func.func @main(%arg0: tensor<82x92xi32>, %arg1: tensor<82x92xi32>, %arg2: tensor<10x40x46x39x21xf32>) -> (tensor<82x92xi32>, tensor<10x40x46x39x21xi1>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<82x92xi32>, tensor<82x92xi32>) -> tensor<82x92xi32>
    %1 = tosa.tanh %arg2 : (tensor<10x40x46x39x21xf32>) -> tensor<10x40x46x39x21xf32>
    %2 = tosa.equal %1, %1 : (tensor<10x40x46x39x21xf32>, tensor<10x40x46x39x21xf32>) -> tensor<10x40x46x39x21xi1>
    %3 = tosa.logical_or %2, %2 : (tensor<10x40x46x39x21xi1>, tensor<10x40x46x39x21xi1>) -> tensor<10x40x46x39x21xi1>
    return %0, %3 : tensor<82x92xi32>, tensor<10x40x46x39x21xi1>
  }
}
