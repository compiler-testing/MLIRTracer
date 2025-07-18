module {
  func.func @main(%arg0: tensor<90x15x46x92x75x89xi8>, %arg1: tensor<90x15x46x1x75x1xi8>, %arg2: tensor<98x23x93x80xf32>, %arg3: tensor<24x69x21x9xf32>, %arg4: tensor<24xf32>) -> (tensor<90x15x46x92x75x89xi1>, tensor<98x94x208x24xf32>, tensor<98x94x208x24xf32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<90x15x46x92x75x89xi8>, tensor<90x15x46x1x75x1xi8>) -> tensor<90x15x46x92x75x89xi8>
    %1 = tosa.sub %0, %0 : (tensor<90x15x46x92x75x89xi8>, tensor<90x15x46x92x75x89xi8>) -> tensor<90x15x46x92x75x89xi8>
    %2 = tosa.greater_equal %1, %0 : (tensor<90x15x46x92x75x89xi8>, tensor<90x15x46x92x75x89xi8>) -> tensor<90x15x46x92x75x89xi1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 98, 94, 208, 24>} : (tensor<98x23x93x80xf32>, tensor<24x69x21x9xf32>, tensor<24xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<98x94x208x24xf32>
    %4 = tosa.logical_or %2, %2 : (tensor<90x15x46x92x75x89xi1>, tensor<90x15x46x92x75x89xi1>) -> tensor<90x15x46x92x75x89xi1>
    %5 = tosa.reciprocal %3 : (tensor<98x94x208x24xf32>) -> tensor<98x94x208x24xf32>
    %6 = tosa.clamp %3 {min_val = 3.300000e+01 : f32, max_val = 7.000000e+01 : f32} : (tensor<98x94x208x24xf32>) -> tensor<98x94x208x24xf32>
    return %4, %5, %6 : tensor<90x15x46x92x75x89xi1>, tensor<98x94x208x24xf32>, tensor<98x94x208x24xf32>
  }
}
