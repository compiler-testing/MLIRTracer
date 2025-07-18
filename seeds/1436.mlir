module {
  func.func @main(%arg0: tensor<24x95x21x9xf32>, %arg1: tensor<69xi8>, %arg2: tensor<1xi8>) -> (tensor<69xi8>, tensor<30x14364x2x1xf32>) {
    %0 = tosa.tanh %arg0 : (tensor<24x95x21x9xf32>) -> tensor<24x95x21x9xf32>
    %1 = tosa.arithmetic_right_shift %arg1, %arg2 {round = true} : (tensor<69xi8>, tensor<1xi8>) -> tensor<69xi8>
    %2 = tosa.tanh %0 : (tensor<24x95x21x9xf32>) -> tensor<24x95x21x9xf32>
    %r_3 = tosa.const_shape {values = dense<[ 30, 7182, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.reshape %2, %r_3 : (tensor<24x95x21x9xf32>, !tosa.shape<4>) -> tensor<30x7182x2x1xf32>
    %4 = tosa.concat %3, %3 {axis = 1 : i32} : (tensor<30x7182x2x1xf32>, tensor<30x7182x2x1xf32>) -> tensor<30x14364x2x1xf32>
    %5 = tosa.log %4 : (tensor<30x14364x2x1xf32>) -> tensor<30x14364x2x1xf32>
    return %1, %5 : tensor<69xi8>, tensor<30x14364x2x1xf32>
  }
}
