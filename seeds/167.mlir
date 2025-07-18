module {
  func.func @main(%arg0: tensor<39xf32>, %arg1: tensor<39xf32>, %arg2: tensor<14x19x94x57xf32>, %arg3: tensor<81x3x66x72xf32>, %arg4: tensor<81xf32>) -> (tensor<39xi1>, tensor<14x24x161x81xf32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<39xf32>, tensor<39xf32>) -> tensor<39xi1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<39xi1>, tensor<39xi1>) -> tensor<39xi1>
    %2 = tosa.logical_xor %1, %1 : (tensor<39xi1>, tensor<39xi1>) -> tensor<39xi1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 14, 24, 161, 81>} : (tensor<14x19x94x57xf32>, tensor<81x3x66x72xf32>, tensor<81xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<14x24x161x81xf32>
    %4 = tosa.sigmoid %3 : (tensor<14x24x161x81xf32>) -> tensor<14x24x161x81xf32>
    return %2, %4 : tensor<39xi1>, tensor<14x24x161x81xf32>
  }
}
