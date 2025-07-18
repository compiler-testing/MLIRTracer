module {
  func.func @main(%arg0: tensor<23x21x43x82xf32>, %arg1: tensor<1x21x1x82xf32>) -> tensor<6x7x3x4xf32> {
    %0 = tosa.pow %arg0, %arg1 : (tensor<23x21x43x82xf32>, tensor<1x21x1x82xf32>) -> tensor<23x21x43x82xf32>
    %s_1_start = tosa.const_shape {values = dense<[ 14, 7, 13, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_1_size = tosa.const_shape {values = dense<[ 6, 7, 3, 4 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<23x21x43x82xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<6x7x3x4xf32>
    return %1 : tensor<6x7x3x4xf32>
  }
}
