module {
  func.func @main(%arg0: tensor<56x81xi16>) -> tensor<112x2xi16> {
    %0 = tosa.reduce_min %arg0 {axis = 1 : i32} : (tensor<56x81xi16>) -> tensor<56x1xi16>
    %t_1 = tosa.const_shape {values = dense<[ 2, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %0, %t_1 : (tensor<56x1xi16>, !tosa.shape<2>) -> tensor<112x2xi16>
    return %1 : tensor<112x2xi16>
  }
}
