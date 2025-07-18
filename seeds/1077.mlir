module {
  func.func @main(%arg0: tensor<44x77x87xi1>, %arg1: tensor<44x77x1xi1>, %arg2: tensor<37x53x70xi32>, %arg3: tensor<37x1x70xi32>) -> (tensor<44x77x174xi1>, tensor<111x106x70xi32>, tensor<37x70xi32>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<44x77x87xi1>, tensor<44x77x1xi1>) -> tensor<44x77x87xi1>
    %1 = tosa.concat %0, %0 {axis = 2 : i32} : (tensor<44x77x87xi1>, tensor<44x77x87xi1>) -> tensor<44x77x174xi1>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<44x77x174xi1>, tensor<44x77x174xi1>) -> tensor<44x77x174xi1>
    %3 = tosa.logical_xor %2, %1 : (tensor<44x77x174xi1>, tensor<44x77x174xi1>) -> tensor<44x77x174xi1>
    %4 = tosa.maximum %arg2, %arg3 : (tensor<37x53x70xi32>, tensor<37x1x70xi32>) -> tensor<37x53x70xi32>
    %5 = tosa.minimum %4, %4 : (tensor<37x53x70xi32>, tensor<37x53x70xi32>) -> tensor<37x53x70xi32>
    %6 = tosa.bitwise_not %4 : (tensor<37x53x70xi32>) -> tensor<37x53x70xi32>
    %t_7 = tosa.const_shape {values = dense<[ 3, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %7 = tosa.tile %6, %t_7 : (tensor<37x53x70xi32>, !tosa.shape<3>) -> tensor<111x106x70xi32>
    %8 = tosa.sub %5, %4 : (tensor<37x53x70xi32>, tensor<37x53x70xi32>) -> tensor<37x53x70xi32>
    %9 = tosa.argmax %8 {axis = 1 : i32} : (tensor<37x53x70xi32>) -> tensor<37x70xi32>
    %10 = tosa.intdiv %7, %7 : (tensor<111x106x70xi32>, tensor<111x106x70xi32>) -> tensor<111x106x70xi32>
    %11 = tosa.sub %9, %9 : (tensor<37x70xi32>, tensor<37x70xi32>) -> tensor<37x70xi32>
    return %3, %10, %11 : tensor<44x77x174xi1>, tensor<111x106x70xi32>, tensor<37x70xi32>
  }
}
