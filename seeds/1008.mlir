module {
  func.func @main(%arg0: tensor<37xi8>, %arg1: tensor<f32>, %arg2: tensor<80xi1>) -> (tensor<1x1xi32>, tensor<1xi1>, tensor<1x1xi32>, tensor<f32>) {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<37xi8>) -> tensor<1xi8>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<1xi8>) -> tensor<1xi8>
    %s_2_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_2_size = tosa.const_shape {values = dense<[ 12 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<1xi8>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<12xi8>
    %3 = tosa.logical_right_shift %2, %2 : (tensor<12xi8>, tensor<12xi8>) -> tensor<12xi8>
    %4 = tosa.reduce_sum %3 {axis = 0 : i32} : (tensor<12xi8>) -> tensor<1xi8>
    %5 = tosa.bitwise_not %4 : (tensor<1xi8>) -> tensor<1xi8>
    %r_6 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.reshape %5, %r_6 : (tensor<1xi8>, !tosa.shape<3>) -> tensor<1x1x1xi8>
    %7 = tosa.reverse %6 {axis = 0 : i32} : (tensor<1x1x1xi8>) -> tensor<1x1x1xi8>
    %8 = tosa.argmax %7 {axis = 1 : i32} : (tensor<1x1x1xi8>) -> tensor<1x1xi32>
    %9 = tosa.sub %8, %8 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
    %10 = tosa.reverse %9 {axis = 1 : i32} : (tensor<1x1xi32>) -> tensor<1x1xi32>
    %11 = tosa.add %10, %10 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
    %12 = tosa.clamp %11 {min_val = -22 : i32, max_val = 45 : i32} : (tensor<1x1xi32>) -> tensor<1x1xi32>
    %13 = tosa.ceil %arg1 : (tensor<f32>) -> tensor<f32>
    %14 = tosa.tanh %13 : (tensor<f32>) -> tensor<f32>
    %15 = tosa.reduce_all %arg2 {axis = 0 : i32} : (tensor<80xi1>) -> tensor<1xi1>
    %16 = tosa.minimum %11, %9 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
    %17 = tosa.tanh %14 : (tensor<f32>) -> tensor<f32>
    return %12, %15, %16, %17 : tensor<1x1xi32>, tensor<1xi1>, tensor<1x1xi32>, tensor<f32>
  }
}
