module {
  func.func @main(%arg0: tensor<95x77xi16>, %arg1: tensor<95x1xi16>, %arg2: tensor<f32>) -> (tensor<1x5xi16>, tensor<285x154xi16>, tensor<i1>, tensor<f32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<95x77xi16>, tensor<95x1xi16>) -> tensor<95x77xi16>
    %1 = tosa.floor %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.reciprocal %1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.floor %2 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.bitwise_and %0, %0 : (tensor<95x77xi16>, tensor<95x77xi16>) -> tensor<95x77xi16>
    %s_5_start = tosa.const_shape {values = dense<[ 83, 47 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_5_size = tosa.const_shape {values = dense<[ 1, 5 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<95x77xi16>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x5xi16>
    %6 = tosa.equal %3, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %t_7 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %7 = tosa.tile %4, %t_7 : (tensor<95x77xi16>, !tosa.shape<2>) -> tensor<285x154xi16>
    %8 = tosa.logical_and %6, %6 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %9 = tosa.sigmoid %1 : (tensor<f32>) -> tensor<f32>
    return %5, %7, %8, %9 : tensor<1x5xi16>, tensor<285x154xi16>, tensor<i1>, tensor<f32>
  }
}
