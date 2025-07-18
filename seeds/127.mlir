module {
  func.func @main(%arg0: tensor<18x52x53x39xf32>, %arg1: tensor<9x86x70x35xf32>, %arg2: tensor<9xf32>, %arg3: tensor<70x8xi8>) -> (tensor<6x1x12x7xf32>, tensor<70x8xi8>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 18, 191, 125, 9>} : (tensor<18x52x53x39xf32>, tensor<9x86x70x35xf32>, tensor<9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<18x191x125x9xf32>
    %s_1_start = tosa.const_shape {values = dense<[ 1, 9, 11, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_1_size = tosa.const_shape {values = dense<[ 6, 1, 12, 7 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<18x191x125x9xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<6x1x12x7xf32>
    %2 = tosa.clz %arg3 : (tensor<70x8xi8>) -> tensor<70x8xi8>
    return %1, %2 : tensor<6x1x12x7xf32>, tensor<70x8xi8>
  }
}
