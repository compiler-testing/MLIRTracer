module {
  func.func @main(%arg0: tensor<45x86x49x53xf32>, %arg1: tensor<77x33x93x45xi16>, %arg2: tensor<1x1x93x45xi16>) -> (tensor<77x33x93x45xi16>, tensor<45x86x49x53xi1>) {
    %0 = tosa.tanh %arg0 : (tensor<45x86x49x53xf32>) -> tensor<45x86x49x53xf32>
    %1 = tosa.logical_right_shift %arg1, %arg2 : (tensor<77x33x93x45xi16>, tensor<1x1x93x45xi16>) -> tensor<77x33x93x45xi16>
    %2 = tosa.greater_equal %0, %0 : (tensor<45x86x49x53xf32>, tensor<45x86x49x53xf32>) -> tensor<45x86x49x53xi1>
    return %1, %2 : tensor<77x33x93x45xi16>, tensor<45x86x49x53xi1>
  }
}
