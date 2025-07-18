module {
  func.func @main(%arg0: tensor<80x65xf32>, %arg1: tensor<29x5xi1>) -> (tensor<80x65xf32>, tensor<87x5xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<80x65xf32>) -> tensor<80x65xf32>
    %1 = tosa.logical_not %arg1 : (tensor<29x5xi1>) -> tensor<29x5xi1>
    %2 = tosa.floor %0 : (tensor<80x65xf32>) -> tensor<80x65xf32>
    %t_3 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.tile %1, %t_3 : (tensor<29x5xi1>, !tosa.shape<2>) -> tensor<87x5xi1>
    return %2, %3 : tensor<80x65xf32>, tensor<87x5xi1>
  }
}
