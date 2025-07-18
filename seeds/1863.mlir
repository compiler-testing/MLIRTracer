module {
  func.func @main(%arg0: tensor<24x41x42x13xf32>, %arg1: tensor<24x41x1x1xf32>, %arg2: tensor<84x32x41x60xi1>) -> (tensor<24x82x126x26xf32>, tensor<84x32x2x2xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<24x41x42x13xf32>, tensor<24x41x1x1xf32>) -> tensor<24x41x42x13xf32>
    %1 = tosa.reduce_any %arg2 {axis = 2 : i32} : (tensor<84x32x41x60xi1>) -> tensor<84x32x1x60xi1>
    %2 = tosa.reduce_max %1 {axis = 3 : i32} : (tensor<84x32x1x60xi1>) -> tensor<84x32x1x1xi1>
    %t_3 = tosa.const_shape {values = dense<[ 1, 2, 3, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.tile %0, %t_3 : (tensor<24x41x42x13xf32>, !tosa.shape<4>) -> tensor<24x82x126x26xf32>
    %4 = tosa.logical_right_shift %2, %2 : (tensor<84x32x1x1xi1>, tensor<84x32x1x1xi1>) -> tensor<84x32x1x1xi1>
    %5 = tosa.concat %4, %4 {axis = 3 : i32} : (tensor<84x32x1x1xi1>, tensor<84x32x1x1xi1>) -> tensor<84x32x1x2xi1>
    %6 = tosa.concat %5, %5 {axis = 2 : i32} : (tensor<84x32x1x2xi1>, tensor<84x32x1x2xi1>) -> tensor<84x32x2x2xi1>
    return %3, %6 : tensor<24x82x126x26xf32>, tensor<84x32x2x2xi1>
  }
}
