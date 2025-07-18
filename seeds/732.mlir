module {
  func.func @main(%arg0: tensor<95x21xi8>, %arg1: tensor<91xi32>, %arg2: tensor<91xi32>, %arg3: tensor<f32>) -> (tensor<91xi32>, tensor<f32>, tensor<190x1xi8>) {
    %t_0 = tosa.const_shape {values = dense<[ 2, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.tile %arg0, %t_0 : (tensor<95x21xi8>, !tosa.shape<2>) -> tensor<190x21xi8>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<91xi32>, tensor<91xi32>) -> tensor<91xi32>
    %2 = tosa.bitwise_and %1, %1 : (tensor<91xi32>, tensor<91xi32>) -> tensor<91xi32>
    %3 = tosa.reciprocal %arg3 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<190x21xi8>) -> tensor<190x1xi8>
    return %2, %3, %4 : tensor<91xi32>, tensor<f32>, tensor<190x1xi8>
  }
}
