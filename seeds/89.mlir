module {
  func.func @main(%arg0: tensor<63x83xi1>) -> tensor<6x5x9xi1> {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<63x83xi1>) -> tensor<1x83xi1>
    %r_1 = tosa.const_shape {values = dense<[ 83, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.reshape %0, %r_1 : (tensor<1x83xi1>, !tosa.shape<3>) -> tensor<83x1x1xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 17, 0, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_2_size = tosa.const_shape {values = dense<[ 6, 5, 9 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<83x1x1xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<6x5x9xi1>
    return %2 : tensor<6x5x9xi1>
  }
}
