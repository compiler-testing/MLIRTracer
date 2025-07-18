module {
  func.func @main(%arg0: tensor<7x41x62xi64>, %arg1: tensor<7x62x85xi64>, %arg2: tensor<41x46x12x93xf32>, %arg3: tensor<65x8x8x16xf32>, %arg4: tensor<65xf32>) -> (tensor<7x41x85xi64>, tensor<41x101x34x65xf32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<7x41x62xi64>, tensor<7x62x85xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<7x41x85xi64>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 41, 101, 34, 65>} : (tensor<41x46x12x93xf32>, tensor<65x8x8x16xf32>, tensor<65xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<41x101x34x65xf32>
    %2 = tosa.sigmoid %1 : (tensor<41x101x34x65xf32>) -> tensor<41x101x34x65xf32>
    return %0, %2 : tensor<7x41x85xi64>, tensor<41x101x34x65xf32>
  }
}
