module {
  func.func @main(%arg0: tensor<58xi16>, %arg1: tensor<f32>, %arg2: tensor<41x4x22xi1>) -> (tensor<58xi16>, tensor<f32>, tensor<41x4x22xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<58xi16>, !tosa.shape<1>) -> tensor<58xi16>
    %1 = tosa.tanh %arg1 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.abs %1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.logical_not %arg2 : (tensor<41x4x22xi1>) -> tensor<41x4x22xi1>
    %4 = tosa.bitwise_not %3 : (tensor<41x4x22xi1>) -> tensor<41x4x22xi1>
    return %0, %2, %4 : tensor<58xi16>, tensor<f32>, tensor<41x4x22xi1>
  }
}
