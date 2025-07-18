module {
  func.func @main(%arg0: tensor<37x41xi64>, %arg1: tensor<37x41xi64>, %arg2: tensor<23x18x46x16xf32>, %arg3: tensor<16x85x38x22xf32>, %arg4: tensor<16xf32>, %arg5: tensor<59x60x47x81x88xi1>, %arg6: tensor<59x60x47x1x88xi1>) -> (tensor<37x41xi64>, tensor<23x121x131x16xf32>, tensor<59x60x47x81x88xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<37x41xi64>, tensor<37x41xi64>) -> tensor<37x41xi64>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 23, 121, 131, 16>} : (tensor<23x18x46x16xf32>, tensor<16x85x38x22xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<23x121x131x16xf32>
    %2 = tosa.logical_or %arg5, %arg6 : (tensor<59x60x47x81x88xi1>, tensor<59x60x47x1x88xi1>) -> tensor<59x60x47x81x88xi1>
    return %0, %1, %2 : tensor<37x41xi64>, tensor<23x121x131x16xf32>, tensor<59x60x47x81x88xi1>
  }
}
