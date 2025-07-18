module {
  func.func @main(%arg0: tensor<1x36x16xi8>, %arg1: tensor<54x61x56x28xf32>, %arg2: tensor<48x43x41x30xf32>, %arg3: tensor<48xf32>) -> (tensor<36x16xi32>, tensor<54x166x155x48xf32>, tensor<54x498x2x144xf32>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<1x36x16xi8>) -> tensor<36x16xi32>
    %1 = tosa.bitwise_and %0, %0 : (tensor<36x16xi32>, tensor<36x16xi32>) -> tensor<36x16xi32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 54, 166, 155, 48>} : (tensor<54x61x56x28xf32>, tensor<48x43x41x30xf32>, tensor<48xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<54x166x155x48xf32>
    %3 = tosa.tanh %2 : (tensor<54x166x155x48xf32>) -> tensor<54x166x155x48xf32>
    %4 = tosa.reduce_min %2 {axis = 2 : i32} : (tensor<54x166x155x48xf32>) -> tensor<54x166x1x48xf32>
    %t_5 = tosa.const_shape {values = dense<[ 1, 3, 2, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.tile %4, %t_5 : (tensor<54x166x1x48xf32>, !tosa.shape<4>) -> tensor<54x498x2x144xf32>
    return %1, %3, %5 : tensor<36x16xi32>, tensor<54x166x155x48xf32>, tensor<54x498x2x144xf32>
  }
}
