module {
  func.func @main(%arg0: tensor<55x61x79xf32>, %arg1: tensor<22xi1>, %arg2: tensor<22xi1>) -> (tensor<55x61x79xf32>, tensor<22xi1>) {
    %0 = tosa.rsqrt %arg0 : (tensor<55x61x79xf32>) -> tensor<55x61x79xf32>
    %1 = tosa.arithmetic_right_shift %arg1, %arg2 {round = true} : (tensor<22xi1>, tensor<22xi1>) -> tensor<22xi1>
    %r_2 = tosa.const_shape {values = dense<[ 22 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.reshape %1, %r_2 : (tensor<22xi1>, !tosa.shape<1>) -> tensor<22xi1>
    %3 = tosa.logical_and %2, %1 : (tensor<22xi1>, tensor<22xi1>) -> tensor<22xi1>
    return %0, %3 : tensor<55x61x79xf32>, tensor<22xi1>
  }
}
