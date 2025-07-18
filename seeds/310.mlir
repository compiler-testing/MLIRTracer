module {
  func.func @main(%arg0: tensor<74x64x63x21x11x73xi8>) -> tensor<4x6x6x11x10x10xi8> {
    %s_0_start = tosa.const_shape {values = dense<[ 8, 51, 21, 10, 1, 34 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_0_size = tosa.const_shape {values = dense<[ 4, 6, 6, 11, 10, 10 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<74x64x63x21x11x73xi8>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<4x6x6x11x10x10xi8>
    return %0 : tensor<4x6x6x11x10x10xi8>
  }
}
