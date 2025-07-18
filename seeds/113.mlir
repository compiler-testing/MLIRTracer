module {
  func.func @main(%arg0: tensor<9x75x19x73x10xi32>, %arg1: tensor<1x1x19x1x1xi32>, %arg2: tensor<f32>, %arg3: tensor<46x47xf32>, %arg4: tensor<1x47xf32>) -> (tensor<f32>, tensor<2x99x60x2xi1>, tensor<138x141xf32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<9x75x19x73x10xi32>, tensor<1x1x19x1x1xi32>) -> tensor<9x75x19x73x10xi1>
    %1 = tosa.rsqrt %arg2 : (tensor<f32>) -> tensor<f32>
    %s_2_start = tosa.const_shape {values = dense<[ 3, 1, 7, 3, 0 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_2_size = tosa.const_shape {values = dense<[ 3, 11, 10, 3, 12 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %2 = tosa.slice %0, %s_2_start, %s_2_size : (tensor<9x75x19x73x10xi1>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<3x11x10x3x12xi1>
    %3 = tosa.minimum %arg3, %arg4 : (tensor<46x47xf32>, tensor<1x47xf32>) -> tensor<46x47xf32>
    %r_4 = tosa.const_shape {values = dense<[ 1, 99, 60, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = tosa.reshape %2, %r_4 : (tensor<3x11x10x3x12xi1>, !tosa.shape<4>) -> tensor<1x99x60x2xi1>
    %5 = tosa.concat %4, %4 {axis = 0 : i32} : (tensor<1x99x60x2xi1>, tensor<1x99x60x2xi1>) -> tensor<2x99x60x2xi1>
    %t_6 = tosa.const_shape {values = dense<[ 3, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.tile %3, %t_6 : (tensor<46x47xf32>, !tosa.shape<2>) -> tensor<138x141xf32>
    return %1, %5, %6 : tensor<f32>, tensor<2x99x60x2xi1>, tensor<138x141xf32>
  }
}
