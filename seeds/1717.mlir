module {
  func.func @main(%arg0: tensor<77x17x21x21x42x19xi8>, %arg1: tensor<77x1x1x21x1x1xi8>, %arg2: tensor<81x36xf32>) -> (tensor<77x17x21x42x42x19xi8>, tensor<243x108xi1>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<77x17x21x21x42x19xi8>, tensor<77x1x1x21x1x1xi8>) -> tensor<77x17x21x21x42x19xi8>
    %1 = tosa.sigmoid %arg2 : (tensor<81x36xf32>) -> tensor<81x36xf32>
    %2 = tosa.concat %0, %0 {axis = 3 : i32} : (tensor<77x17x21x21x42x19xi8>, tensor<77x17x21x21x42x19xi8>) -> tensor<77x17x21x42x42x19xi8>
    %3 = tosa.identity %1 : (tensor<81x36xf32>) -> tensor<81x36xf32>
    %4 = tosa.greater_equal %3, %3 : (tensor<81x36xf32>, tensor<81x36xf32>) -> tensor<81x36xi1>
    %t_5 = tosa.const_shape {values = dense<[ 3, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.tile %4, %t_5 : (tensor<81x36xi1>, !tosa.shape<2>) -> tensor<243x108xi1>
    return %2, %5 : tensor<77x17x21x42x42x19xi8>, tensor<243x108xi1>
  }
}
