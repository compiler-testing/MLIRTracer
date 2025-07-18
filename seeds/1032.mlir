module {
  func.func @main(%arg0: tensor<75xi1>, %arg1: tensor<1xi1>, %arg2: tensor<f32>, %arg3: tensor<77x77x37x97x98xi64>, %arg4: tensor<77x1x1x97x1xi64>) -> (tensor<1xi1>, tensor<f32>, tensor<f32>, tensor<3x3x9x12x5xi64>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<75xi1>, tensor<1xi1>) -> tensor<75xi1>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<75xi1>) -> tensor<75xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 57 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_2_size = tosa.const_shape {values = dense<[ 10 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<75xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<10xi1>
    %3 = tosa.logical_or %2, %2 : (tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
    %4 = tosa.add %3, %2 : (tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
    %5 = tosa.reduce_sum %4 {axis = 0 : i32} : (tensor<10xi1>) -> tensor<1xi1>
    %6 = tosa.reduce_any %5 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.rsqrt %arg2 : (tensor<f32>) -> tensor<f32>
    %8 = tosa.logical_left_shift %6, %6 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %9 = tosa.reduce_min %8 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %10 = tosa.maximum %arg3, %arg4 : (tensor<77x77x37x97x98xi64>, tensor<77x1x1x97x1xi64>) -> tensor<77x77x37x97x98xi64>
    %11 = tosa.reduce_any %9 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %s_12_start = tosa.const_shape {values = dense<[ 36, 44, 28, 63, 68 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_12_size = tosa.const_shape {values = dense<[ 3, 3, 9, 12, 5 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %12 = tosa.slice %10, %s_12_start, %s_12_size : (tensor<77x77x37x97x98xi64>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<3x3x9x12x5xi64>
    %13 = tosa.logical_right_shift %11, %9 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %14 = tosa.floor %7 : (tensor<f32>) -> tensor<f32>
    %15 = tosa.tanh %7 : (tensor<f32>) -> tensor<f32>
    %16 = tosa.reciprocal %14 : (tensor<f32>) -> tensor<f32>
    %17 = tosa.add %12, %12 : (tensor<3x3x9x12x5xi64>, tensor<3x3x9x12x5xi64>) -> tensor<3x3x9x12x5xi64>
    return %13, %15, %16, %17 : tensor<1xi1>, tensor<f32>, tensor<f32>, tensor<3x3x9x12x5xi64>
  }
}
