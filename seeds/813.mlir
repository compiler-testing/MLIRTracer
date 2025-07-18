module {
  func.func @main(%arg0: tensor<39x31xf32>, %arg1: tensor<10x89x66x77x70x39xi8>, %arg2: tensor<10x89x1x77x70x1xi8>, %arg3: tensor<i1>, %arg4: tensor<i1>) -> (tensor<i1>, tensor<78x62xf32>, tensor<10x89x66x77x70x39xi1>) {
    %0 = tosa.floor %arg0 : (tensor<39x31xf32>) -> tensor<39x31xf32>
    %1 = tosa.abs %0 : (tensor<39x31xf32>) -> tensor<39x31xf32>
    %2 = tosa.logical_left_shift %arg1, %arg2 : (tensor<10x89x66x77x70x39xi8>, tensor<10x89x1x77x70x1xi8>) -> tensor<10x89x66x77x70x39xi8>
    %3 = tosa.bitwise_or %2, %2 : (tensor<10x89x66x77x70x39xi8>, tensor<10x89x66x77x70x39xi8>) -> tensor<10x89x66x77x70x39xi8>
    %4 = tosa.logical_xor %arg3, %arg4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %t_5 = tosa.const_shape {values = dense<[ 2, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.tile %1, %t_5 : (tensor<39x31xf32>, !tosa.shape<2>) -> tensor<78x62xf32>
    %6 = tosa.greater_equal %2, %3 : (tensor<10x89x66x77x70x39xi8>, tensor<10x89x66x77x70x39xi8>) -> tensor<10x89x66x77x70x39xi1>
    return %4, %5, %6 : tensor<i1>, tensor<78x62xf32>, tensor<10x89x66x77x70x39xi1>
  }
}
