module {
  func.func @main(%arg0: tensor<48x45xi8>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<f32>, tensor<3x45xi8>, tensor<1x45xi8>, tensor<3x1xi8>, tensor<i32>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<48x45xi8>) -> tensor<1x45xi8>
    %t_1 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %0, %t_1 : (tensor<1x45xi8>, !tosa.shape<2>) -> tensor<3x45xi8>
    %2 = tosa.reciprocal %arg1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.tanh %2 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.bitwise_and %1, %1 : (tensor<3x45xi8>, tensor<3x45xi8>) -> tensor<3x45xi8>
    %5 = tosa.reverse %4 {axis = 0 : i32} : (tensor<3x45xi8>) -> tensor<3x45xi8>
    %6 = tosa.bitwise_and %4, %1 : (tensor<3x45xi8>, tensor<3x45xi8>) -> tensor<3x45xi8>
    %7 = tosa.reduce_sum %4 {axis = 0 : i32} : (tensor<3x45xi8>) -> tensor<1x45xi8>
    %8 = tosa.intdiv %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %9 = tosa.reduce_max %6 {axis = 1 : i32} : (tensor<3x45xi8>) -> tensor<3x1xi8>
    %10 = tosa.bitwise_xor %8, %8 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %3, %5, %7, %9, %10 : tensor<f32>, tensor<3x45xi8>, tensor<1x45xi8>, tensor<3x1xi8>, tensor<i32>
  }
}
