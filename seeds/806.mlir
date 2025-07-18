module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<11x55xi8>, %arg3: tensor<11x91xi8>) -> (tensor<i1>, tensor<11x146xi8>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.identity %1 : (tensor<i1>) -> tensor<i1>
    %3 = tosa.concat %arg2, %arg3 {axis = 1 : i32} : (tensor<11x55xi8>, tensor<11x91xi8>) -> tensor<11x146xi8>
    return %2, %3 : tensor<i1>, tensor<11x146xi8>
  }
}
