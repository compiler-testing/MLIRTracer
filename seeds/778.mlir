module {
  func.func @main(%arg0: tensor<99xi1>, %arg1: tensor<1xi1>, %arg2: tensor<3x21x33x75xf32>, %arg3: tensor<57x49x33x63xf32>, %arg4: tensor<57xf32>, %arg5: tensor<99x79x21x28xf32>, %arg6: tensor<99xf32>) -> (tensor<99xi1>, tensor<3x262x221x99xf32>, tensor<3x91x100x57xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<99xi1>, tensor<1xi1>) -> tensor<99xi1>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 3, 91, 100, 57>} : (tensor<3x21x33x75xf32>, tensor<57x49x33x63xf32>, tensor<57xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3x91x100x57xf32>
    %2 = tosa.add %1, %1 : (tensor<3x91x100x57xf32>, tensor<3x91x100x57xf32>) -> tensor<3x91x100x57xf32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %2, %arg5, %arg6, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 3, 262, 221, 99>} : (tensor<3x91x100x57xf32>, tensor<99x79x21x28xf32>, tensor<99xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3x262x221x99xf32>
    %4 = tosa.maximum %2, %2 : (tensor<3x91x100x57xf32>, tensor<3x91x100x57xf32>) -> tensor<3x91x100x57xf32>
    %5 = tosa.greater %4, %4 : (tensor<3x91x100x57xf32>, tensor<3x91x100x57xf32>) -> tensor<3x91x100x57xi1>
    return %0, %3, %5 : tensor<99xi1>, tensor<3x262x221x99xf32>, tensor<3x91x100x57xi1>
  }
}
