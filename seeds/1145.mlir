module {
  func.func @main(%arg0: tensor<60xi16>) -> tensor<8xi16> {
    %s_0_start = tosa.const_shape {values = dense<[ 7 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_0_size = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<60xi16>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<8xi16>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<8xi16>) -> tensor<8xi16>
    return %1 : tensor<8xi16>
  }
}
