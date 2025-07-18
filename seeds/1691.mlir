module {
  func.func @main(%arg0: tensor<14x20x47xi16>) -> tensor<1x40x141xi16> {
    %t_0 = tosa.const_shape {values = dense<[ 2, 2, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<14x20x47xi16>, !tosa.shape<3>) -> tensor<28x40x141xi16>
    %1 = tosa.bitwise_and %0, %0 : (tensor<28x40x141xi16>, tensor<28x40x141xi16>) -> tensor<28x40x141xi16>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<28x40x141xi16>) -> tensor<1x40x141xi16>
    return %2 : tensor<1x40x141xi16>
  }
}
