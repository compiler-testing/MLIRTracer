module {
  func.func @main(%arg0: tensor<75x52xf32>, %arg1: tensor<67x82xi8>, %arg2: tensor<67x82xi8>, %arg3: tensor<88x28x98xi1>) -> (tensor<67x82xi8>, tensor<1x28x98xi1>, tensor<150x52xf32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<75x52xf32>) -> tensor<75x52xf32>
    %1 = tosa.bitwise_xor %arg1, %arg2 : (tensor<67x82xi8>, tensor<67x82xi8>) -> tensor<67x82xi8>
    %2 = tosa.reduce_any %arg3 {axis = 0 : i32} : (tensor<88x28x98xi1>) -> tensor<1x28x98xi1>
    %t_3 = tosa.const_shape {values = dense<[ 2, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.tile %0, %t_3 : (tensor<75x52xf32>, !tosa.shape<2>) -> tensor<150x52xf32>
    return %1, %2, %3 : tensor<67x82xi8>, tensor<1x28x98xi1>, tensor<150x52xf32>
  }
}
