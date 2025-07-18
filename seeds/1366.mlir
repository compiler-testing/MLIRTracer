module {
  func.func @main(%arg0: tensor<100x60x6x86xi16>, %arg1: tensor<53x17x94xi1>) -> (tensor<9x12x1x9xi16>, tensor<53x1x94xi1>) {
    %s_0_start = tosa.const_shape {values = dense<[ 80, 37, 5, 62 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_0_size = tosa.const_shape {values = dense<[ 9, 12, 1, 9 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<100x60x6x86xi16>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<9x12x1x9xi16>
    %1 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<53x17x94xi1>) -> tensor<53x1x94xi1>
    return %0, %1 : tensor<9x12x1x9xi16>, tensor<53x1x94xi1>
  }
}
