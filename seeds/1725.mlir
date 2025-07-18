module {
  func.func @main(%arg0: tensor<95xi16>, %arg1: tensor<95xi16>, %arg2: tensor<29x2x97x42xf32>, %arg3: tensor<57x40x48x66xf32>, %arg4: tensor<57xf32>) -> (tensor<1xi16>, tensor<29x43x147x57xf32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<95xi16>, tensor<95xi16>) -> tensor<95xi16>
    %1 = tosa.clamp %0 {min_val = 20 : i16, max_val = 63 : i16} : (tensor<95xi16>) -> tensor<95xi16>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<95xi16>) -> tensor<1xi16>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 29, 43, 147, 57>} : (tensor<29x2x97x42xf32>, tensor<57x40x48x66xf32>, tensor<57xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<29x43x147x57xf32>
    return %2, %3 : tensor<1xi16>, tensor<29x43x147x57xf32>
  }
}
