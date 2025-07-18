module {
  func.func @main(%arg0: tensor<27x98xf32>, %arg1: tensor<12x17x15x42x92x86xi16>, %arg2: tensor<12x1x15x42x92x1xi16>, %arg3: tensor<9x2x55x66xf32>, %arg4: tensor<4x86x91x14xf32>, %arg5: tensor<4xf32>) -> (tensor<12x17x15x42x92x86xi16>, tensor<27x98xf32>, tensor<9x91x201x4xf32>) {
    %0 = tosa.floor %arg0 : (tensor<27x98xf32>) -> tensor<27x98xf32>
    %1 = tosa.bitwise_xor %arg1, %arg2 : (tensor<12x17x15x42x92x86xi16>, tensor<12x1x15x42x92x1xi16>) -> tensor<12x17x15x42x92x86xi16>
    %in_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.negate %0, %in_zp_2, %out_zp_2 : (tensor<27x98xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<27x98xf32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 9, 91, 201, 4>} : (tensor<9x2x55x66xf32>, tensor<4x86x91x14xf32>, tensor<4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<9x91x201x4xf32>
    return %1, %2, %3 : tensor<12x17x15x42x92x86xi16>, tensor<27x98xf32>, tensor<9x91x201x4xf32>
  }
}
