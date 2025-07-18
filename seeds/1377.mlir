module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<46x23xi16>, %arg2: tensor<1x1xi16>) -> (tensor<f32>, tensor<i1>, tensor<f32>, tensor<46x1xi16>, tensor<11x6xi16>) {
    %0 = tosa.tanh %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.ceil %0 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.sub %1, %0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = tosa.bitwise_or %arg1, %arg2 : (tensor<46x23xi16>, tensor<1x1xi16>) -> tensor<46x23xi16>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<46x23xi16>) -> tensor<1x23xi16>
    %5 = tosa.reduce_sum %4 {axis = 1 : i32} : (tensor<1x23xi16>) -> tensor<1x1xi16>
    %s_6_start = tosa.const_shape {values = dense<[ 0, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_6_size = tosa.const_shape {values = dense<[ 11, 6 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.slice %5, %s_6_start, %s_6_size : (tensor<1x1xi16>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<11x6xi16>
    %7 = tosa.greater %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = tosa.log %0 : (tensor<f32>) -> tensor<f32>
    %9 = tosa.logical_right_shift %6, %6 : (tensor<11x6xi16>, tensor<11x6xi16>) -> tensor<11x6xi16>
    %10 = tosa.floor %8 : (tensor<f32>) -> tensor<f32>
    %11 = tosa.reduce_min %3 {axis = 1 : i32} : (tensor<46x23xi16>) -> tensor<46x1xi16>
    %12 = tosa.logical_left_shift %9, %6 : (tensor<11x6xi16>, tensor<11x6xi16>) -> tensor<11x6xi16>
    return %2, %7, %10, %11, %12 : tensor<f32>, tensor<i1>, tensor<f32>, tensor<46x1xi16>, tensor<11x6xi16>
  }
}
