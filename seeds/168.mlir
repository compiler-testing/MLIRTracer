module {
  func.func @main(%arg0: tensor<72x74x83xi32>, %arg1: tensor<72x74x83xi32>, %arg2: tensor<72x4x35xf32>) -> (tensor<3x8x7xf32>, tensor<72x74x83xi1>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<72x74x83xi32>, tensor<72x74x83xi32>) -> tensor<72x74x83xi1>
    %1 = tosa.floor %arg2 : (tensor<72x4x35xf32>) -> tensor<72x4x35xf32>
    %s_2_start = tosa.const_shape {values = dense<[ 58, 0, 14 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_2_size = tosa.const_shape {values = dense<[ 3, 8, 7 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<72x4x35xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<3x8x7xf32>
    %3 = tosa.bitwise_xor %0, %0 : (tensor<72x74x83xi1>, tensor<72x74x83xi1>) -> tensor<72x74x83xi1>
    return %2, %3 : tensor<3x8x7xf32>, tensor<72x74x83xi1>
  }
}
