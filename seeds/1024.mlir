module {
  func.func @main(%arg0: tensor<34x30x81x13xi8>, %arg1: tensor<1x1x1x1xi8>, %arg2: tensor<89x70xf32>) -> (tensor<34x30x81x1xi8>, tensor<89x1xf32>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<34x30x81x13xi8>, tensor<1x1x1x1xi8>) -> tensor<34x30x81x13xi8>
    %1 = tosa.reciprocal %arg2 : (tensor<89x70xf32>) -> tensor<89x70xf32>
    %2 = tosa.reduce_sum %0 {axis = 3 : i32} : (tensor<34x30x81x13xi8>) -> tensor<34x30x81x1xi8>
    %3 = tosa.identity %2 : (tensor<34x30x81x1xi8>) -> tensor<34x30x81x1xi8>
    %4 = tosa.reduce_min %1 {axis = 1 : i32} : (tensor<89x70xf32>) -> tensor<89x1xf32>
    %5 = tosa.minimum %3, %3 : (tensor<34x30x81x1xi8>, tensor<34x30x81x1xi8>) -> tensor<34x30x81x1xi8>
    %r_6 = tosa.const_shape {values = dense<[ 89, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.reshape %4, %r_6 : (tensor<89x1xf32>, !tosa.shape<2>) -> tensor<89x1xf32>
    return %5, %6 : tensor<34x30x81x1xi8>, tensor<89x1xf32>
  }
}
