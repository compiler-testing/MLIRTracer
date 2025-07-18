module {
  func.func @main(%arg0: tensor<73xf32>, %arg1: tensor<63x28xi1>) -> (tensor<1xf32>, tensor<1x28xi1>, tensor<8x5xi1>) {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<73xf32>) -> tensor<1xf32>
    %1 = tosa.clamp %0 {min_val = 5.500000e+01 : f32, max_val = 9.100000e+01 : f32} : (tensor<1xf32>) -> tensor<1xf32>
    %2 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<63x28xi1>) -> tensor<1x28xi1>
    %3 = tosa.clz %2 : (tensor<1x28xi1>) -> tensor<1x28xi1>
    %s_4_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_4_size = tosa.const_shape {values = dense<[ 8, 5 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.slice %2, %s_4_start, %s_4_size : (tensor<1x28xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<8x5xi1>
    %5 = tosa.logical_left_shift %4, %4 : (tensor<8x5xi1>, tensor<8x5xi1>) -> tensor<8x5xi1>
    %6 = tosa.logical_left_shift %5, %5 : (tensor<8x5xi1>, tensor<8x5xi1>) -> tensor<8x5xi1>
    return %1, %3, %6 : tensor<1xf32>, tensor<1x28xi1>, tensor<8x5xi1>
  }
}
