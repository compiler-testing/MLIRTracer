module {
  func.func @main(%arg0: tensor<1x83x98x2x82x98xi8>, %arg1: tensor<f32>) -> (tensor<22x2x11x4x3x1xi8>, tensor<f32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 0, 1, 1, 0, 1, 1 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_0_size = tosa.const_shape {values = dense<[ 11, 2, 11, 4, 3, 1 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<1x83x98x2x82x98xi8>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<11x2x11x4x3x1xi8>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<11x2x11x4x3x1xi8>, tensor<11x2x11x4x3x1xi8>) -> tensor<22x2x11x4x3x1xi8>
    %2 = tosa.exp %arg1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.sigmoid %2 : (tensor<f32>) -> tensor<f32>
    return %1, %3 : tensor<22x2x11x4x3x1xi8>, tensor<f32>
  }
}
