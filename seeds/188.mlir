module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<26x51x28x68x71xf32>) -> (tensor<1x1xi8>, tensor<26x51x28x68x71xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %1 = tosa.bitwise_and %0, %0 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %r_3 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.reshape %2, %r_3 : (tensor<i8>, !tosa.shape<2>) -> tensor<1x1xi8>
    %4 = tosa.exp %arg2 : (tensor<26x51x28x68x71xf32>) -> tensor<26x51x28x68x71xf32>
    %5 = tosa.ceil %4 : (tensor<26x51x28x68x71xf32>) -> tensor<26x51x28x68x71xf32>
    return %3, %5 : tensor<1x1xi8>, tensor<26x51x28x68x71xf32>
  }
}
