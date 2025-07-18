module {
  func.func @main(%arg0: tensor<32x99x7x98x66xi1>, %arg1: tensor<1x99x7x98x1xi1>, %arg2: tensor<53x12x5x52x1x44xf32>, %arg3: tensor<1x12x5x52x1x1xf32>, %arg4: tensor<75x67xf32>) -> (tensor<32x99x7x98x66xi1>, tensor<3x4x4x1x9x6xf32>, tensor<67xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<32x99x7x98x66xi1>, tensor<1x99x7x98x1xi1>) -> tensor<32x99x7x98x66xi1>
    %1 = tosa.add %0, %0 : (tensor<32x99x7x98x66xi1>, tensor<32x99x7x98x66xi1>) -> tensor<32x99x7x98x66xi1>
    %2 = tosa.pow %arg2, %arg3 : (tensor<53x12x5x52x1x44xf32>, tensor<1x12x5x52x1x1xf32>) -> tensor<53x12x5x52x1x44xf32>
    %s_3_start = tosa.const_shape {values = dense<[ 10, 8, 1, 41, 0, 26 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_3_size = tosa.const_shape {values = dense<[ 3, 4, 4, 1, 9, 6 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<53x12x5x52x1x44xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<3x4x4x1x9x6xf32>
    %4 = tosa.argmax %arg4 {axis = 0 : i32} : (tensor<75x67xf32>) -> tensor<67xi32>
    %5 = tosa.greater %4, %4 : (tensor<67xi32>, tensor<67xi32>) -> tensor<67xi1>
    return %1, %3, %5 : tensor<32x99x7x98x66xi1>, tensor<3x4x4x1x9x6xf32>, tensor<67xi1>
  }
}
