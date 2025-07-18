module {
  func.func @main(%arg0: tensor<8x43xi1>, %arg1: tensor<1x43xi1>, %arg2: tensor<39xf32>, %arg3: tensor<1xf32>) -> (tensor<8x86xi1>, tensor<39xf32>, tensor<11xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<8x43xi1>, tensor<1x43xi1>) -> tensor<8x43xi1>
    %1 = tosa.pow %arg2, %arg3 : (tensor<39xf32>, tensor<1xf32>) -> tensor<39xf32>
    %2 = tosa.concat %0, %0 {axis = 1 : i32} : (tensor<8x43xi1>, tensor<8x43xi1>) -> tensor<8x86xi1>
    %3 = tosa.reciprocal %1 : (tensor<39xf32>) -> tensor<39xf32>
    %s_4_start = tosa.const_shape {values = dense<[ 12 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_4_size = tosa.const_shape {values = dense<[ 11 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.slice %1, %s_4_start, %s_4_size : (tensor<39xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<11xf32>
    return %2, %3, %4 : tensor<8x86xi1>, tensor<39xf32>, tensor<11xf32>
  }
}
