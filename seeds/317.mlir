module {
  func.func @main(%arg0: tensor<50x50xi16>, %arg1: tensor<1x1xi16>) -> tensor<50x100xi16> {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<50x50xi16>, tensor<1x1xi16>) -> tensor<50x50xi16>
    %t_1 = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %0, %t_1 : (tensor<50x50xi16>, !tosa.shape<2>) -> tensor<50x100xi16>
    %2 = tosa.bitwise_not %1 : (tensor<50x100xi16>) -> tensor<50x100xi16>
    return %2 : tensor<50x100xi16>
  }
}
