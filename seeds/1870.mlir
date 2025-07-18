module {
  func.func @main(%arg0: tensor<8x4x22x50x72xi8>, %arg1: tensor<5x2xi32>, %arg2: tensor<80x13x69x2x36xf32>, %arg3: tensor<83x88x53xi32>, %arg4: tensor<83x88x53xi32>) -> (tensor<8x4x22x50x72xi8>, tensor<80x13x69x2x36xf32>, tensor<83x88x53xi32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<10xindex>} : () -> !tosa.shape<10>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<8x4x22x50x72xi8>, !tosa.shape<10>, tensor<1xi8>) -> tensor<8x4x22x50x72xi8>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<8x4x22x50x72xi8>, tensor<8x4x22x50x72xi8>) -> tensor<8x4x22x50x72xi8>
    %2 = tosa.sub %1, %1 : (tensor<8x4x22x50x72xi8>, tensor<8x4x22x50x72xi8>) -> tensor<8x4x22x50x72xi8>
    %3 = tosa.floor %arg2 : (tensor<80x13x69x2x36xf32>) -> tensor<80x13x69x2x36xf32>
    %4 = tosa.intdiv %arg3, %arg4 : (tensor<83x88x53xi32>, tensor<83x88x53xi32>) -> tensor<83x88x53xi32>
    %5 = tosa.minimum %4, %4 : (tensor<83x88x53xi32>, tensor<83x88x53xi32>) -> tensor<83x88x53xi32>
    return %2, %3, %5 : tensor<8x4x22x50x72xi8>, tensor<80x13x69x2x36xf32>, tensor<83x88x53xi32>
  }
}
