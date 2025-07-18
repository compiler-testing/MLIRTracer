module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<i16>, %arg2: tensor<12xi1>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> (tensor<i16>, tensor<3xi1>, tensor<i1>, tensor<5xi1>, tensor<22xi1>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %1 = tosa.clamp %0 {min_val = -6 : i16, max_val = 30 : i16} : (tensor<i16>) -> tensor<i16>
    %2 = tosa.reduce_product %arg2 {axis = 0 : i32} : (tensor<12xi1>) -> tensor<1xi1>
    %3 = tosa.pow %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %t_4 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.tile %2, %t_4 : (tensor<1xi1>, !tosa.shape<1>) -> tensor<3xi1>
    %5 = tosa.logical_or %4, %4 : (tensor<3xi1>, tensor<3xi1>) -> tensor<3xi1>
    %6 = tosa.logical_right_shift %2, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.ceil %3 : (tensor<f32>) -> tensor<f32>
    %8 = tosa.abs %6 : (tensor<1xi1>) -> tensor<1xi1>
    %s_9_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_9_size = tosa.const_shape {values = dense<[ 11 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %9 = tosa.slice %6, %s_9_start, %s_9_size : (tensor<1xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<11xi1>
    %10 = tosa.add %7, %7 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = tosa.logical_and %9, %9 : (tensor<11xi1>, tensor<11xi1>) -> tensor<11xi1>
    %12 = tosa.equal %10, %10 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %13 = tosa.reduce_all %8 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %s_14_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_14_size = tosa.const_shape {values = dense<[ 5 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %14 = tosa.slice %13, %s_14_start, %s_14_size : (tensor<1xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<5xi1>
    %15 = tosa.concat %11, %9 {axis = 0 : i32} : (tensor<11xi1>, tensor<11xi1>) -> tensor<22xi1>
    %16 = tosa.sub %15, %15 : (tensor<22xi1>, tensor<22xi1>) -> tensor<22xi1>
    return %1, %5, %12, %14, %16 : tensor<i16>, tensor<3xi1>, tensor<i1>, tensor<5xi1>, tensor<22xi1>
  }
}
