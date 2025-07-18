module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<70x7x29xi16>) -> (tensor<i1>, tensor<210x7x2xi16>, tensor<140x7x1xi16>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.reduce_min %arg2 {axis = 2 : i32} : (tensor<70x7x29xi16>) -> tensor<70x7x1xi16>
    %t_2 = tosa.const_shape {values = dense<[ 3, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.tile %1, %t_2 : (tensor<70x7x1xi16>, !tosa.shape<3>) -> tensor<210x7x2xi16>
    %3 = tosa.add %2, %2 : (tensor<210x7x2xi16>, tensor<210x7x2xi16>) -> tensor<210x7x2xi16>
    %4 = tosa.bitwise_or %1, %1 : (tensor<70x7x1xi16>, tensor<70x7x1xi16>) -> tensor<70x7x1xi16>
    %5 = tosa.arithmetic_right_shift %4, %1 {round = true} : (tensor<70x7x1xi16>, tensor<70x7x1xi16>) -> tensor<70x7x1xi16>
    %6 = tosa.concat %5, %4 {axis = 0 : i32} : (tensor<70x7x1xi16>, tensor<70x7x1xi16>) -> tensor<140x7x1xi16>
    return %0, %3, %6 : tensor<i1>, tensor<210x7x2xi16>, tensor<140x7x1xi16>
  }
}
