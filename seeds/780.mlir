module {
  func.func @main(%arg0: tensor<56x15x72x26xi1>, %arg1: tensor<100x43x2x77xi8>, %arg2: tensor<100x43x2x77xi8>) -> (tensor<43x2x77xi32>, tensor<3x12x6x1xi1>) {
    %s_0_start = tosa.const_shape {values = dense<[ 53, 3, 11, 12 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_0_size = tosa.const_shape {values = dense<[ 3, 12, 6, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<56x15x72x26xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x12x6x1xi1>
    %1 = tosa.greater %arg1, %arg2 : (tensor<100x43x2x77xi8>, tensor<100x43x2x77xi8>) -> tensor<100x43x2x77xi1>
    %2 = tosa.logical_or %1, %1 : (tensor<100x43x2x77xi1>, tensor<100x43x2x77xi1>) -> tensor<100x43x2x77xi1>
    %3 = tosa.add %2, %1 : (tensor<100x43x2x77xi1>, tensor<100x43x2x77xi1>) -> tensor<100x43x2x77xi1>
    %4 = tosa.argmax %3 {axis = 0 : i32} : (tensor<100x43x2x77xi1>) -> tensor<43x2x77xi32>
    %5 = tosa.logical_xor %0, %0 : (tensor<3x12x6x1xi1>, tensor<3x12x6x1xi1>) -> tensor<3x12x6x1xi1>
    return %4, %5 : tensor<43x2x77xi32>, tensor<3x12x6x1xi1>
  }
}
