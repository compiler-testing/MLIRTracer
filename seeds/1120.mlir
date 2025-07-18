module {
  func.func @main(%arg0: tensor<96x52x51x45xi8>, %arg1: tensor<1x1x51x1xi8>, %arg2: tensor<32x94x80x54x77x50xi32>, %arg3: tensor<32x1x1x54x77x1xi32>) -> (tensor<96x52x51x45xi1>, tensor<32x94x80x54x77x50xi32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<96x52x51x45xi8>, tensor<1x1x51x1xi8>) -> tensor<96x52x51x45xi1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<96x52x51x45xi1>, tensor<96x52x51x45xi1>) -> tensor<96x52x51x45xi1>
    %2 = tosa.maximum %arg2, %arg3 : (tensor<32x94x80x54x77x50xi32>, tensor<32x1x1x54x77x1xi32>) -> tensor<32x94x80x54x77x50xi32>
    %3 = tosa.identity %2 : (tensor<32x94x80x54x77x50xi32>) -> tensor<32x94x80x54x77x50xi32>
    %4 = tosa.logical_left_shift %1, %1 : (tensor<96x52x51x45xi1>, tensor<96x52x51x45xi1>) -> tensor<96x52x51x45xi1>
    %5 = tosa.minimum %3, %2 : (tensor<32x94x80x54x77x50xi32>, tensor<32x94x80x54x77x50xi32>) -> tensor<32x94x80x54x77x50xi32>
    return %4, %5 : tensor<96x52x51x45xi1>, tensor<32x94x80x54x77x50xi32>
  }
}
