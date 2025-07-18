module {
  func.func @main(%arg0: tensor<32xi32>, %arg1: tensor<89x28x88x59xi1>, %arg2: tensor<89x1x88x1xi1>) -> (tensor<44x73514x4x1xi1>, tensor<2xi32>) {
    %t_0 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<32xi32>, !tosa.shape<1>) -> tensor<96xi32>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<89x28x88x59xi1>, tensor<89x1x88x1xi1>) -> tensor<89x28x88x59xi1>
    %r_2 = tosa.const_shape {values = dense<[ 44, 73514, 4, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.reshape %1, %r_2 : (tensor<89x28x88x59xi1>, !tosa.shape<4>) -> tensor<44x73514x4x1xi1>
    %3 = tosa.logical_right_shift %2, %2 : (tensor<44x73514x4x1xi1>, tensor<44x73514x4x1xi1>) -> tensor<44x73514x4x1xi1>
    %4 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<96xi32>) -> tensor<1xi32>
    %5 = tosa.identity %3 : (tensor<44x73514x4x1xi1>) -> tensor<44x73514x4x1xi1>
    %t_6 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.tile %4, %t_6 : (tensor<1xi32>, !tosa.shape<1>) -> tensor<2xi32>
    return %5, %6 : tensor<44x73514x4x1xi1>, tensor<2xi32>
  }
}
