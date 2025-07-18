module {
  func.func @main(%arg0: tensor<95x45x58x40x61x13xi8>, %arg1: tensor<1x1x1x1x1x13xi8>) -> tensor<6x10x12x11x12x5xi8> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<95x45x58x40x61x13xi8>, tensor<1x1x1x1x1x13xi8>) -> tensor<95x45x58x40x61x13xi8>
    %s_1_start = tosa.const_shape {values = dense<[ 85, 35, 46, 29, 25, 8 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_1_size = tosa.const_shape {values = dense<[ 6, 10, 12, 11, 12, 5 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<95x45x58x40x61x13xi8>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<6x10x12x11x12x5xi8>
    return %1 : tensor<6x10x12x11x12x5xi8>
  }
}
