module {
  func.func @main(%arg0: tensor<31xf32>, %arg1: tensor<79x34xi16>, %arg2: tensor<1x1xi16>, %arg3: tensor<58x95x1x65xf32>, %arg4: tensor<20x2x36x13xf32>, %arg5: tensor<20xf32>) -> (tensor<31xf32>, tensor<79x34xi16>, tensor<58x192x40x20xf32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<31xf32>) -> tensor<31xf32>
    %1 = tosa.bitwise_xor %arg1, %arg2 : (tensor<79x34xi16>, tensor<1x1xi16>) -> tensor<79x34xi16>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 58, 192, 40, 20>} : (tensor<58x95x1x65xf32>, tensor<20x2x36x13xf32>, tensor<20xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<58x192x40x20xf32>
    return %0, %1, %2 : tensor<31xf32>, tensor<79x34xi16>, tensor<58x192x40x20xf32>
  }
}
