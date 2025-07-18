module {
  func.func @main(%arg0: tensor<26xi8>, %arg1: tensor<f32>) -> (tensor<i1>, tensor<1xi8>, tensor<1xi8>, tensor<f32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 19 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_0_size = tosa.const_shape {values = dense<[ 7 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<26xi8>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<7xi8>
    %1 = tosa.exp %arg1 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.exp %1 : (tensor<f32>) -> tensor<f32>
    %t_3 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.tile %0, %t_3 : (tensor<7xi8>, !tosa.shape<1>) -> tensor<7xi8>
    %4 = tosa.greater_equal %2, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = tosa.tanh %1 : (tensor<f32>) -> tensor<f32>
    %6 = tosa.reverse %3 {axis = 0 : i32} : (tensor<7xi8>) -> tensor<7xi8>
    %7 = tosa.reduce_min %6 {axis = 0 : i32} : (tensor<7xi8>) -> tensor<1xi8>
    %8 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<7xi8>) -> tensor<1xi8>
    %9 = tosa.add %5, %2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %4, %7, %8, %9 : tensor<i1>, tensor<1xi8>, tensor<1xi8>, tensor<f32>
  }
}
