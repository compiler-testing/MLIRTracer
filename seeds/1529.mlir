module {
  func.func @main(%arg0: tensor<28x32x63x65x71x78xi32>, %arg1: tensor<1x1x1x1x1x78xi32>, %arg2: tensor<77x68x34xf32>) -> (tensor<28x32x63x65x71x78xi1>, tensor<77x68x34xf32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<28x32x63x65x71x78xi32>, tensor<1x1x1x1x1x78xi32>) -> tensor<28x32x63x65x71x78xi1>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<28x32x63x65x71x78xi1>, tensor<28x32x63x65x71x78xi1>) -> tensor<28x32x63x65x71x78xi1>
    %2 = tosa.bitwise_or %1, %1 : (tensor<28x32x63x65x71x78xi1>, tensor<28x32x63x65x71x78xi1>) -> tensor<28x32x63x65x71x78xi1>
    %3 = tosa.tanh %arg2 : (tensor<77x68x34xf32>) -> tensor<77x68x34xf32>
    return %2, %3 : tensor<28x32x63x65x71x78xi1>, tensor<77x68x34xf32>
  }
}
