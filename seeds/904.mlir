module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<44x4x86x17xi8>) -> (tensor<f32>, tensor<12x12x1x1xi8>, tensor<44x4x86x1xi8>) {
    %0 = tosa.tanh %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.reduce_sum %arg1 {axis = 3 : i32} : (tensor<44x4x86x17xi8>) -> tensor<44x4x86x1xi8>
    %2 = tosa.add %1, %1 : (tensor<44x4x86x1xi8>, tensor<44x4x86x1xi8>) -> tensor<44x4x86x1xi8>
    %s_3_start = tosa.const_shape {values = dense<[ 29, 0, 31, 0 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_3_size = tosa.const_shape {values = dense<[ 12, 12, 1, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<44x4x86x1xi8>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<12x12x1x3xi8>
    %4 = tosa.bitwise_or %3, %3 : (tensor<12x12x1x3xi8>, tensor<12x12x1x3xi8>) -> tensor<12x12x1x3xi8>
    %5 = tosa.abs %2 : (tensor<44x4x86x1xi8>) -> tensor<44x4x86x1xi8>
    %6 = tosa.reduce_max %4 {axis = 3 : i32} : (tensor<12x12x1x3xi8>) -> tensor<12x12x1x1xi8>
    %7 = tosa.clamp %5 {min_val = 4 : i8, max_val = 29 : i8} : (tensor<44x4x86x1xi8>) -> tensor<44x4x86x1xi8>
    return %0, %6, %7 : tensor<f32>, tensor<12x12x1x1xi8>, tensor<44x4x86x1xi8>
  }
}
