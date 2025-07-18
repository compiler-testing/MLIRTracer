module {
  func.func @main(%arg0: tensor<23x13x58xi16>, %arg1: tensor<23x13x58xi16>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<i1>, tensor<69x13x58xi16>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<23x13x58xi16>, tensor<23x13x58xi16>) -> tensor<23x13x58xi16>
    %1 = tosa.greater %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %t_2 = tosa.const_shape {values = dense<[ 3, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.tile %0, %t_2 : (tensor<23x13x58xi16>, !tosa.shape<3>) -> tensor<69x13x58xi16>
    return %1, %2 : tensor<i1>, tensor<69x13x58xi16>
  }
}
