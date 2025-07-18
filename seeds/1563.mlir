module {
  func.func @main(%arg0: tensor<87xi8>, %arg1: tensor<87xi8>, %arg2: tensor<i1>, %arg3: tensor<i1>) -> (tensor<i1>, tensor<1xi8>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<87xi8>, tensor<87xi8>) -> tensor<87xi8>
    %1 = tosa.sub %0, %0 : (tensor<87xi8>, tensor<87xi8>) -> tensor<87xi8>
    %2 = tosa.clz %1 : (tensor<87xi8>) -> tensor<87xi8>
    %3 = tosa.logical_xor %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.clz %2 : (tensor<87xi8>) -> tensor<87xi8>
    %5 = tosa.reduce_min %4 {axis = 0 : i32} : (tensor<87xi8>) -> tensor<1xi8>
    return %3, %5 : tensor<i1>, tensor<1xi8>
  }
}
