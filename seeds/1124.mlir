module {
  func.func @main(%arg0: tensor<41x13x54xi32>, %arg1: tensor<41x13x1xi32>, %arg2: tensor<42x33x62x36xf32>, %arg3: tensor<15x24x77x1xf32>, %arg4: tensor<15xf32>) -> (tensor<123x26x108xi32>, tensor<42x60x141x15xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<41x13x54xi32>, tensor<41x13x1xi32>) -> tensor<41x13x54xi32>
    %t_1 = tosa.const_shape {values = dense<[ 3, 2, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %0, %t_1 : (tensor<41x13x54xi32>, !tosa.shape<3>) -> tensor<123x26x108xi32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 42, 60, 141, 15>} : (tensor<42x33x62x36xf32>, tensor<15x24x77x1xf32>, tensor<15xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<42x60x141x15xf32>
    %3 = tosa.minimum %2, %2 : (tensor<42x60x141x15xf32>, tensor<42x60x141x15xf32>) -> tensor<42x60x141x15xf32>
    return %1, %3 : tensor<123x26x108xi32>, tensor<42x60x141x15xf32>
  }
}
