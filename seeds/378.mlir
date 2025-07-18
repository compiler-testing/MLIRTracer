module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<57x74xf32>, %arg2: tensor<29x19xi1>, %arg3: tensor<29x19xi1>) -> (tensor<9x7xf32>, tensor<f32>, tensor<29x1xi1>) {
    %0 = tosa.tanh %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.reduce_max %arg1 {axis = 1 : i32} : (tensor<57x74xf32>) -> tensor<57x1xf32>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<29x19xi1>, tensor<29x19xi1>) -> tensor<29x19xi1>
    %3 = tosa.sigmoid %1 : (tensor<57x1xf32>) -> tensor<57x1xf32>
    %s_4_start = tosa.const_shape {values = dense<[ 48, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_4_size = tosa.const_shape {values = dense<[ 9, 7 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.slice %3, %s_4_start, %s_4_size : (tensor<57x1xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<9x7xf32>
    %5 = tosa.reciprocal %0 : (tensor<f32>) -> tensor<f32>
    %6 = tosa.logical_xor %2, %2 : (tensor<29x19xi1>, tensor<29x19xi1>) -> tensor<29x19xi1>
    %7 = tosa.reduce_product %6 {axis = 1 : i32} : (tensor<29x19xi1>) -> tensor<29x1xi1>
    return %4, %5, %7 : tensor<9x7xf32>, tensor<f32>, tensor<29x1xi1>
  }
}
