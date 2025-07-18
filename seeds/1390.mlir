module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<i16>, %arg2: tensor<22xf32>, %arg3: tensor<1xf32>) -> (tensor<i16>, tensor<5xi1>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %1 = tosa.greater %arg2, %arg3 : (tensor<22xf32>, tensor<1xf32>) -> tensor<22xi1>
    %2 = tosa.logical_left_shift %0, %0 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %s_3_start = tosa.const_shape {values = dense<[ 7 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_3_size = tosa.const_shape {values = dense<[ 5 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.slice %1, %s_3_start, %s_3_size : (tensor<22xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<5xi1>
    return %2, %3 : tensor<i16>, tensor<5xi1>
  }
}
