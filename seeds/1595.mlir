module {
  func.func @main(%arg0: tensor<43xi16>, %arg1: tensor<19x48xi64>, %arg2: tensor<19x1xi64>, %arg3: tensor<f32>, %arg4: tensor<i1>, %arg5: tensor<98xi1>) -> (tensor<8xi16>, tensor<f32>, tensor<48xi32>, tensor<f32>, tensor<1xi1>, tensor<19x48xi64>, tensor<i1>) {
    %t_0 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<43xi16>, !tosa.shape<1>) -> tensor<86xi16>
    %1 = tosa.maximum %arg1, %arg2 : (tensor<19x48xi64>, tensor<19x1xi64>) -> tensor<19x48xi64>
    %2 = tosa.tanh %arg3 : (tensor<f32>) -> tensor<f32>
    %s_3_start = tosa.const_shape {values = dense<[ 15 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_3_size = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.slice %0, %s_3_start, %s_3_size : (tensor<86xi16>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<8xi16>
    %4 = tosa.logical_not %arg4 : (tensor<i1>) -> tensor<i1>
    %5 = tosa.rsqrt %2 : (tensor<f32>) -> tensor<f32>
    %6 = tosa.tanh %5 : (tensor<f32>) -> tensor<f32>
    %7 = tosa.argmax %1 {axis = 0 : i32} : (tensor<19x48xi64>) -> tensor<48xi32>
    %8 = tosa.exp %2 : (tensor<f32>) -> tensor<f32>
    %9 = tosa.reduce_all %arg5 {axis = 0 : i32} : (tensor<98xi1>) -> tensor<1xi1>
    %10 = tosa.clz %1 : (tensor<19x48xi64>) -> tensor<19x48xi64>
    %11 = tosa.bitwise_xor %4, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %3, %6, %7, %8, %9, %10, %11 : tensor<8xi16>, tensor<f32>, tensor<48xi32>, tensor<f32>, tensor<1xi1>, tensor<19x48xi64>, tensor<i1>
  }
}
