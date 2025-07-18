module {
  func.func @main(%arg0: tensor<75x72x3xi32>, %arg1: tensor<76x87x2xf32>) -> (tensor<9x6x3xi32>, tensor<76x87x2xf32>) {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<75x72x3xi32>) -> tensor<1x72x3xi32>
    %s_1_start = tosa.const_shape {values = dense<[ 0, 1, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_1_size = tosa.const_shape {values = dense<[ 9, 6, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<1x72x3xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<9x6x3xi32>
    %2 = tosa.floor %arg1 : (tensor<76x87x2xf32>) -> tensor<76x87x2xf32>
    return %1, %2 : tensor<9x6x3xi32>, tensor<76x87x2xf32>
  }
}
