module {
  func.func @main(%arg0: tensor<37x74xi16>, %arg1: tensor<22x55x1x22xf32>, %arg2: tensor<66x61x56x32xf32>, %arg3: tensor<66xf32>) -> (tensor<111x148xi16>, tensor<22x173x58x66xf32>, tensor<14569368xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.tile %arg0, %t_0 : (tensor<37x74xi16>, !tosa.shape<2>) -> tensor<111x148xi16>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 22, 173, 58, 66>} : (tensor<22x55x1x22xf32>, tensor<66x61x56x32xf32>, tensor<66xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<22x173x58x66xf32>
    %2 = tosa.pow %1, %1 : (tensor<22x173x58x66xf32>, tensor<22x173x58x66xf32>) -> tensor<22x173x58x66xf32>
    %r_3 = tosa.const_shape {values = dense<[ 14569368 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.reshape %1, %r_3 : (tensor<22x173x58x66xf32>, !tosa.shape<1>) -> tensor<14569368xf32>
    return %0, %2, %3 : tensor<111x148xi16>, tensor<22x173x58x66xf32>, tensor<14569368xf32>
  }
}
