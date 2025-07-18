module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<77x17xi64>, %arg2: tensor<54x63x77x85x73xi1>, %arg3: tensor<54x1x77x1x1xi1>) -> (tensor<f32>, tensor<77x17xi64>, tensor<54x63x77x85x73xi1>) {
    %0 = tosa.floor %arg0 : (tensor<f32>) -> tensor<f32>
    %t_1 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %arg1, %t_1 : (tensor<77x17xi64>, !tosa.shape<2>) -> tensor<77x17xi64>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<54x63x77x85x73xi1>, tensor<54x1x77x1x1xi1>) -> tensor<54x63x77x85x73xi1>
    return %0, %1, %2 : tensor<f32>, tensor<77x17xi64>, tensor<54x63x77x85x73xi1>
  }
}
