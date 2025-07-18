module {
  func.func @main(%arg0: tensor<77x47x6x5x69xf32>, %arg1: tensor<77x47x6x5x69xf32>, %arg2: tensor<2x36x94x42x29xi8>, %arg3: tensor<1x36x94x1x29xi8>) -> (tensor<2x36x94x42x29xi1>, tensor<77x47x6x5x69xf32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<77x47x6x5x69xf32>, tensor<77x47x6x5x69xf32>) -> tensor<77x47x6x5x69xf32>
    %1 = tosa.logical_left_shift %arg2, %arg3 : (tensor<2x36x94x42x29xi8>, tensor<1x36x94x1x29xi8>) -> tensor<2x36x94x42x29xi8>
    %2 = tosa.greater_equal %1, %1 : (tensor<2x36x94x42x29xi8>, tensor<2x36x94x42x29xi8>) -> tensor<2x36x94x42x29xi1>
    %3 = tosa.floor %0 : (tensor<77x47x6x5x69xf32>) -> tensor<77x47x6x5x69xf32>
    %4 = tosa.maximum %3, %0 : (tensor<77x47x6x5x69xf32>, tensor<77x47x6x5x69xf32>) -> tensor<77x47x6x5x69xf32>
    return %2, %4 : tensor<2x36x94x42x29xi1>, tensor<77x47x6x5x69xf32>
  }
}
