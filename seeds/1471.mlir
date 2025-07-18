module {
  func.func @main(%arg0: tensor<83x9x52xi8>) -> tensor<3x12x8xi8> {
    %s_0_start = tosa.const_shape {values = dense<[ 4, 0, 44 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_0_size = tosa.const_shape {values = dense<[ 3, 12, 8 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<83x9x52xi8>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<3x12x8xi8>
    %1 = tosa.add %0, %0 : (tensor<3x12x8xi8>, tensor<3x12x8xi8>) -> tensor<3x12x8xi8>
    return %1 : tensor<3x12x8xi8>
  }
}
