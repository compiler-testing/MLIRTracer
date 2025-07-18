module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<37x18x62xi1>, %arg2: tensor<54x81x82xi8>, %arg3: tensor<54x1x82xi8>) -> (tensor<f32>, tensor<1x1x62xi1>, tensor<1x11x11xi1>) {
    %0 = tosa.floor %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.reduce_min %arg1 {axis = 1 : i32} : (tensor<37x18x62xi1>) -> tensor<37x1x62xi1>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<37x1x62xi1>) -> tensor<37x1x62xi1>
    %3 = tosa.sub %1, %2 : (tensor<37x1x62xi1>, tensor<37x1x62xi1>) -> tensor<37x1x62xi1>
    %4 = tosa.logical_xor %3, %3 : (tensor<37x1x62xi1>, tensor<37x1x62xi1>) -> tensor<37x1x62xi1>
    %5 = tosa.abs %4 : (tensor<37x1x62xi1>) -> tensor<37x1x62xi1>
    %6 = tosa.reduce_max %5 {axis = 0 : i32} : (tensor<37x1x62xi1>) -> tensor<1x1x62xi1>
    %7 = tosa.maximum %arg2, %arg3 : (tensor<54x81x82xi8>, tensor<54x1x82xi8>) -> tensor<54x81x82xi8>
    %s_8_start = tosa.const_shape {values = dense<[ 1, 35, 6 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_8_size = tosa.const_shape {values = dense<[ 1, 11, 11 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %8 = tosa.slice %7, %s_8_start, %s_8_size : (tensor<54x81x82xi8>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x11x11xi8>
    %9 = tosa.greater_equal %8, %8 : (tensor<1x11x11xi8>, tensor<1x11x11xi8>) -> tensor<1x11x11xi1>
    return %0, %6, %9 : tensor<f32>, tensor<1x1x62xi1>, tensor<1x11x11xi1>
  }
}
