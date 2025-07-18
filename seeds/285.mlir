module {
  func.func @main(%arg0: tensor<98xf32>, %arg1: tensor<37xf32>, %arg2: tensor<77x84x95x26xi1>, %arg3: tensor<1x84x1x1xi1>) -> (tensor<77x84x95x26xi1>, tensor<135xf32>, tensor<77x84x95x26xi1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<98xf32>, tensor<37xf32>) -> tensor<135xf32>
    %1 = tosa.minimum %0, %0 : (tensor<135xf32>, tensor<135xf32>) -> tensor<135xf32>
    %2 = tosa.sub %1, %0 : (tensor<135xf32>, tensor<135xf32>) -> tensor<135xf32>
    %3 = tosa.logical_left_shift %arg2, %arg3 : (tensor<77x84x95x26xi1>, tensor<1x84x1x1xi1>) -> tensor<77x84x95x26xi1>
    %4 = tosa.logical_xor %3, %3 : (tensor<77x84x95x26xi1>, tensor<77x84x95x26xi1>) -> tensor<77x84x95x26xi1>
    %5 = tosa.minimum %2, %2 : (tensor<135xf32>, tensor<135xf32>) -> tensor<135xf32>
    %6 = tosa.clz %3 : (tensor<77x84x95x26xi1>) -> tensor<77x84x95x26xi1>
    return %4, %5, %6 : tensor<77x84x95x26xi1>, tensor<135xf32>, tensor<77x84x95x26xi1>
  }
}
