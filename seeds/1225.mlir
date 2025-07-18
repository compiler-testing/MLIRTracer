module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<8x100x8x33x66xf32>, %arg2: tensor<8x100x8x33x66xf32>, %arg3: tensor<10xi1>) -> (tensor<f32>, tensor<8x100x8x33x66xf32>, tensor<1xi1>, tensor<10xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.minimum %arg1, %arg2 : (tensor<8x100x8x33x66xf32>, tensor<8x100x8x33x66xf32>) -> tensor<8x100x8x33x66xf32>
    %2 = tosa.logical_not %arg3 : (tensor<10xi1>) -> tensor<10xi1>
    %3 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<10xi1>) -> tensor<1xi1>
    %4 = tosa.arithmetic_right_shift %2, %2 {round = false} : (tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_5_size = tosa.const_shape {values = dense<[ 10 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<10xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<10xi1>
    return %0, %1, %3, %5 : tensor<f32>, tensor<8x100x8x33x66xf32>, tensor<1xi1>, tensor<10xi1>
  }
}
