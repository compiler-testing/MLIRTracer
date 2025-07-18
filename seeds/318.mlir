module {
  func.func @main(%arg0: tensor<11xi16>, %arg1: tensor<53x8x8x82x97xf32>) -> (tensor<3xi16>, tensor<53x8x8x82x97xf32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_0_size = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<11xi16>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xi16>
    %1 = tosa.exp %arg1 : (tensor<53x8x8x82x97xf32>) -> tensor<53x8x8x82x97xf32>
    return %0, %1 : tensor<3xi16>, tensor<53x8x8x82x97xf32>
  }
}
