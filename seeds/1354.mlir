module {
  func.func @main(%arg0: tensor<82x15xi8>, %arg1: tensor<3xi1>, %arg2: tensor<14x2x67x16x14x31xf32>) -> (tensor<82x15xi8>, tensor<1xi1>, tensor<14x2x67x16x14x31xf32>) {
    %0 = tosa.identity %arg0 : (tensor<82x15xi8>) -> tensor<82x15xi8>
    %1 = tosa.abs %0 : (tensor<82x15xi8>) -> tensor<82x15xi8>
    %2 = tosa.bitwise_not %1 : (tensor<82x15xi8>) -> tensor<82x15xi8>
    %3 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<3xi1>) -> tensor<1xi1>
    %4 = tosa.bitwise_and %3, %3 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_5_size = tosa.const_shape {values = dense<[ 12 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<1xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<12xi1>
    %6 = tosa.reduce_min %5 {axis = 0 : i32} : (tensor<12xi1>) -> tensor<1xi1>
    %7 = tosa.tanh %arg2 : (tensor<14x2x67x16x14x31xf32>) -> tensor<14x2x67x16x14x31xf32>
    return %2, %6, %7 : tensor<82x15xi8>, tensor<1xi1>, tensor<14x2x67x16x14x31xf32>
  }
}
