module {
  func.func @main(%arg0: tensor<15x29x89x84x85x35xf32>, %arg1: tensor<77x55xi1>, %arg2: tensor<77x55xi1>, %arg3: tensor<4x27x94xi32>, %arg4: tensor<4x1x1xi32>) -> (tensor<15x29x89x84x85x35xf32>, tensor<77x55xi1>, tensor<77x55xi1>, tensor<4x27x94xi1>, tensor<4x27x94xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<15x29x89x84x85x35xf32>) -> tensor<15x29x89x84x85x35xf32>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<77x55xi1>, tensor<77x55xi1>) -> tensor<77x55xi1>
    %2 = tosa.intdiv %arg3, %arg4 : (tensor<4x27x94xi32>, tensor<4x1x1xi32>) -> tensor<4x27x94xi32>
    %3 = tosa.logical_not %1 : (tensor<77x55xi1>) -> tensor<77x55xi1>
    %4 = tosa.equal %2, %2 : (tensor<4x27x94xi32>, tensor<4x27x94xi32>) -> tensor<4x27x94xi1>
    %5 = tosa.reverse %1 {axis = 1 : i32} : (tensor<77x55xi1>) -> tensor<77x55xi1>
    %6 = tosa.arithmetic_right_shift %5, %5 {round = true} : (tensor<77x55xi1>, tensor<77x55xi1>) -> tensor<77x55xi1>
    %7 = tosa.clz %4 : (tensor<4x27x94xi1>) -> tensor<4x27x94xi1>
    %8 = tosa.greater %2, %2 : (tensor<4x27x94xi32>, tensor<4x27x94xi32>) -> tensor<4x27x94xi1>
    return %0, %3, %6, %7, %8 : tensor<15x29x89x84x85x35xf32>, tensor<77x55xi1>, tensor<77x55xi1>, tensor<4x27x94xi1>, tensor<4x27x94xi1>
  }
}
