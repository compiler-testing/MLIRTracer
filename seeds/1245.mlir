module {
  func.func @main(%arg0: tensor<87x50x58x81x68xi8>, %arg1: tensor<87x50x1x1x68xi8>, %arg2: tensor<11x32x60xi8>, %arg3: tensor<11x32x1xi8>, %arg4: tensor<53xf32>) -> (tensor<87x50x58x81x68xi1>, tensor<53xf32>, tensor<8x9x9xi1>, tensor<1x9x9xi8>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<87x50x58x81x68xi8>, tensor<87x50x1x1x68xi8>) -> tensor<87x50x58x81x68xi1>
    %1 = tosa.maximum %arg2, %arg3 : (tensor<11x32x60xi8>, tensor<11x32x1xi8>) -> tensor<11x32x60xi8>
    %2 = tosa.tanh %arg4 : (tensor<53xf32>) -> tensor<53xf32>
    %s_3_start = tosa.const_shape {values = dense<[ 3, 6, 8 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_3_size = tosa.const_shape {values = dense<[ 8, 9, 9 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.slice %1, %s_3_start, %s_3_size : (tensor<11x32x60xi8>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<8x9x9xi8>
    %4 = tosa.pow %2, %2 : (tensor<53xf32>, tensor<53xf32>) -> tensor<53xf32>
    %5 = tosa.greater %3, %3 : (tensor<8x9x9xi8>, tensor<8x9x9xi8>) -> tensor<8x9x9xi1>
    %6 = tosa.logical_left_shift %3, %3 : (tensor<8x9x9xi8>, tensor<8x9x9xi8>) -> tensor<8x9x9xi8>
    %7 = tosa.reduce_product %6 {axis = 0 : i32} : (tensor<8x9x9xi8>) -> tensor<1x9x9xi8>
    return %0, %4, %5, %7 : tensor<87x50x58x81x68xi1>, tensor<53xf32>, tensor<8x9x9xi1>, tensor<1x9x9xi8>
  }
}
