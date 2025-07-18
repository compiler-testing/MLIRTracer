module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<4x64x87xf32>) -> (tensor<i1>, tensor<4x192x174xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %t_1 = tosa.const_shape {values = dense<[ 1, 3, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %arg2, %t_1 : (tensor<4x64x87xf32>, !tosa.shape<3>) -> tensor<4x192x174xf32>
    %2 = tosa.add %1, %1 : (tensor<4x192x174xf32>, tensor<4x192x174xf32>) -> tensor<4x192x174xf32>
    %3 = tosa.bitwise_xor %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.greater %2, %1 : (tensor<4x192x174xf32>, tensor<4x192x174xf32>) -> tensor<4x192x174xi1>
    return %3, %4 : tensor<i1>, tensor<4x192x174xi1>
  }
}
