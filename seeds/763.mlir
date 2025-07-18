module {
  func.func @main(%arg0: tensor<100x37x3xi8>, %arg1: tensor<3x2xi32>, %arg2: tensor<99x32x60x78xf32>, %arg3: tensor<29x77x69x89xf32>, %arg4: tensor<29xf32>) -> (tensor<100x37x3xi8>, tensor<99x141x132x29xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<100x37x3xi8>, !tosa.shape<6>, tensor<1xi8>) -> tensor<100x37x3xi8>
    %1 = tosa.bitwise_and %0, %0 : (tensor<100x37x3xi8>, tensor<100x37x3xi8>) -> tensor<100x37x3xi8>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<100x37x3xi8>, tensor<100x37x3xi8>) -> tensor<100x37x3xi8>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 99, 141, 132, 29>} : (tensor<99x32x60x78xf32>, tensor<29x77x69x89xf32>, tensor<29xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<99x141x132x29xf32>
    return %2, %3 : tensor<100x37x3xi8>, tensor<99x141x132x29xf32>
  }
}
