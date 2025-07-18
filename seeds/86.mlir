module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<1x77x73xi32>) -> (tensor<f32>, tensor<5x12xi32>, tensor<f32>, tensor<2x154x146xi1>, tensor<1x154x73xi1>) {
    %0 = tosa.abs %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.sigmoid %0 : (tensor<f32>) -> tensor<f32>
    %t_2 = tosa.const_shape {values = dense<[ 2, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.tile %arg1, %t_2 : (tensor<1x77x73xi32>, !tosa.shape<3>) -> tensor<2x154x73xi32>
    %3 = tosa.intdiv %2, %2 : (tensor<2x154x73xi32>, tensor<2x154x73xi32>) -> tensor<2x154x73xi32>
    %4 = tosa.logical_left_shift %2, %3 : (tensor<2x154x73xi32>, tensor<2x154x73xi32>) -> tensor<2x154x73xi32>
    %5 = tosa.greater_equal %4, %2 : (tensor<2x154x73xi32>, tensor<2x154x73xi32>) -> tensor<2x154x73xi1>
    %6 = tosa.argmax %3 {axis = 0 : i32} : (tensor<2x154x73xi32>) -> tensor<154x73xi32>
    %s_7_start = tosa.const_shape {values = dense<[ 64, 15 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_7_size = tosa.const_shape {values = dense<[ 5, 12 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %7 = tosa.slice %6, %s_7_start, %s_7_size : (tensor<154x73xi32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<5x12xi32>
    %8 = tosa.floor %0 : (tensor<f32>) -> tensor<f32>
    %9 = tosa.logical_or %5, %5 : (tensor<2x154x73xi1>, tensor<2x154x73xi1>) -> tensor<2x154x73xi1>
    %10 = tosa.exp %8 : (tensor<f32>) -> tensor<f32>
    %11 = tosa.bitwise_and %5, %5 : (tensor<2x154x73xi1>, tensor<2x154x73xi1>) -> tensor<2x154x73xi1>
    %12 = tosa.concat %9, %9 {axis = 2 : i32} : (tensor<2x154x73xi1>, tensor<2x154x73xi1>) -> tensor<2x154x146xi1>
    %13 = tosa.reduce_any %11 {axis = 0 : i32} : (tensor<2x154x73xi1>) -> tensor<1x154x73xi1>
    return %1, %7, %10, %12, %13 : tensor<f32>, tensor<5x12xi32>, tensor<f32>, tensor<2x154x146xi1>, tensor<1x154x73xi1>
  }
}
