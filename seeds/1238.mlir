module {
  func.func @main(%arg0: tensor<83x10x67xi1>, %arg1: tensor<1x1x67xi1>, %arg2: tensor<25x44x31x78xf32>, %arg3: tensor<87x88x71x62xf32>, %arg4: tensor<87xf32>) -> (tensor<83x10x67xi1>, tensor<25x177x134x87xf32>, tensor<6x12x1xi32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<83x10x67xi1>, tensor<1x1x67xi1>) -> tensor<83x10x67xi1>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 25, 177, 134, 87>} : (tensor<25x44x31x78xf32>, tensor<87x88x71x62xf32>, tensor<87xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<25x177x134x87xf32>
    %2 = tosa.reverse %0 {axis = 0 : i32} : (tensor<83x10x67xi1>) -> tensor<83x10x67xi1>
    %3 = tosa.argmax %1 {axis = 1 : i32} : (tensor<25x177x134x87xf32>) -> tensor<25x134x87xi32>
    %4 = tosa.floor %1 : (tensor<25x177x134x87xf32>) -> tensor<25x177x134x87xf32>
    %s_5_start = tosa.const_shape {values = dense<[ 7, 14, 18 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_5_size = tosa.const_shape {values = dense<[ 6, 12, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.slice %3, %s_5_start, %s_5_size : (tensor<25x134x87xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<6x12x3xi32>
    %6 = tosa.reduce_product %5 {axis = 2 : i32} : (tensor<6x12x3xi32>) -> tensor<6x12x1xi32>
    %7 = tosa.clz %6 : (tensor<6x12x1xi32>) -> tensor<6x12x1xi32>
    return %2, %4, %7 : tensor<83x10x67xi1>, tensor<25x177x134x87xf32>, tensor<6x12x1xi32>
  }
}
