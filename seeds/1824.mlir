module {
  func.func @main(%arg0: tensor<29xi16>, %arg1: tensor<29xi16>, %arg2: tensor<74xf32>) -> (tensor<74xf32>, tensor<2xi16>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<29xi16>, tensor<29xi16>) -> tensor<29xi16>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<29xi16>) -> tensor<1xi16>
    %t_2 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.tile %1, %t_2 : (tensor<1xi16>, !tosa.shape<1>) -> tensor<2xi16>
    %3 = tosa.bitwise_and %2, %2 : (tensor<2xi16>, tensor<2xi16>) -> tensor<2xi16>
    %4 = tosa.exp %arg2 : (tensor<74xf32>) -> tensor<74xf32>
    %5 = tosa.abs %3 : (tensor<2xi16>) -> tensor<2xi16>
    return %4, %5 : tensor<74xf32>, tensor<2xi16>
  }
}
