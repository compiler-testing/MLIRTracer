module {
  func.func @main(%arg0: tensor<38x34x41x58xf32>, %arg1: tensor<19x34x41x58xf32>, %arg2: tensor<76xi8>, %arg3: tensor<76xi8>) -> (tensor<57x34x41x58xf32>, tensor<1xi8>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<38x34x41x58xf32>, tensor<19x34x41x58xf32>) -> tensor<57x34x41x58xf32>
    %1 = tosa.bitwise_xor %arg2, %arg3 : (tensor<76xi8>, tensor<76xi8>) -> tensor<76xi8>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<76xi8>) -> tensor<1xi8>
    return %0, %2 : tensor<57x34x41x58xf32>, tensor<1xi8>
  }
}
