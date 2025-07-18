module {
  func.func @main(%arg0: tensor<74x53x65xi1>, %arg1: tensor<74x1x65xi1>, %arg2: tensor<1x89x38x20x30x24xf32>) -> (tensor<148x106x130xi1>, tensor<1x89x38x20x30x24xf32>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<74x53x65xi1>, tensor<74x1x65xi1>) -> tensor<74x53x65xi1>
    %1 = tosa.sub %0, %0 : (tensor<74x53x65xi1>, tensor<74x53x65xi1>) -> tensor<74x53x65xi1>
    %t_2 = tosa.const_shape {values = dense<[ 2, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.tile %1, %t_2 : (tensor<74x53x65xi1>, !tosa.shape<3>) -> tensor<148x106x65xi1>
    %3 = tosa.clz %2 : (tensor<148x106x65xi1>) -> tensor<148x106x65xi1>
    %4 = tosa.concat %3, %2 {axis = 2 : i32} : (tensor<148x106x65xi1>, tensor<148x106x65xi1>) -> tensor<148x106x130xi1>
    %5 = tosa.sigmoid %arg2 : (tensor<1x89x38x20x30x24xf32>) -> tensor<1x89x38x20x30x24xf32>
    return %4, %5 : tensor<148x106x130xi1>, tensor<1x89x38x20x30x24xf32>
  }
}
