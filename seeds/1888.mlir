module {
  func.func @main(%arg0: tensor<77x31x32x98x96xi1>, %arg1: tensor<77x31x32x1x96xi1>, %arg2: tensor<56x94xi32>, %arg3: tensor<56x94xi32>) -> (tensor<56x94xi32>, tensor<77x31x32x98x96xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<77x31x32x98x96xi1>, tensor<77x31x32x1x96xi1>) -> tensor<77x31x32x98x96xi1>
    %1 = tosa.minimum %arg2, %arg3 : (tensor<56x94xi32>, tensor<56x94xi32>) -> tensor<56x94xi32>
    %2 = tosa.bitwise_xor %0, %0 : (tensor<77x31x32x98x96xi1>, tensor<77x31x32x98x96xi1>) -> tensor<77x31x32x98x96xi1>
    return %1, %2 : tensor<56x94xi32>, tensor<77x31x32x98x96xi1>
  }
}
