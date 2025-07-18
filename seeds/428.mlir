module {
  func.func @main(%arg0: tensor<54x68x66xf32>, %arg1: tensor<84x93x86x54x57x97xi1>, %arg2: tensor<1x1x1x54x1x97xi1>) -> (tensor<1x136x66xf32>, tensor<84x93x86x54x57x97xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 2, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<54x68x66xf32>, !tosa.shape<3>) -> tensor<108x136x66xf32>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<84x93x86x54x57x97xi1>, tensor<1x1x1x54x1x97xi1>) -> tensor<84x93x86x54x57x97xi1>
    %2 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<108x136x66xf32>) -> tensor<1x136x66xf32>
    %3 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<84x93x86x54x57x97xi1>, tensor<84x93x86x54x57x97xi1>) -> tensor<84x93x86x54x57x97xi1>
    return %2, %3 : tensor<1x136x66xf32>, tensor<84x93x86x54x57x97xi1>
  }
}
