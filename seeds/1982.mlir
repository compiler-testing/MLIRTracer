module {
  func.func @main(%arg0: tensor<97x23x63x98x96x81xi8>) -> tensor<11x2x7x4x12x10xi8> {
    %s_0_start = tosa.const_shape {values = dense<[ 68, 21, 13, 4, 84, 12 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_0_size = tosa.const_shape {values = dense<[ 11, 2, 7, 4, 12, 10 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<97x23x63x98x96x81xi8>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<11x2x7x4x12x10xi8>
    return %0 : tensor<11x2x7x4x12x10xi8>
  }
}
