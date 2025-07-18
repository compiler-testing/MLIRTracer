module {
  func.func @main(%arg0: tensor<12x94x23x53xf32>, %arg1: tensor<1x92x57x75xf32>, %arg2: tensor<1xf32>, %arg3: tensor<49x28x60x96xi1>) -> (tensor<49x28x1x96xi1>, tensor<12x188x83x1xf32>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 12, 188, 83, 1>} : (tensor<12x94x23x53xf32>, tensor<1x92x57x75xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x188x83x1xf32>
    %1 = tosa.reciprocal %0 : (tensor<12x188x83x1xf32>) -> tensor<12x188x83x1xf32>
    %2 = tosa.reduce_any %arg3 {axis = 2 : i32} : (tensor<49x28x60x96xi1>) -> tensor<49x28x1x96xi1>
    %3 = tosa.maximum %1, %0 : (tensor<12x188x83x1xf32>, tensor<12x188x83x1xf32>) -> tensor<12x188x83x1xf32>
    %4 = tosa.clamp %3 {min_val = -5.000000e+01 : f32, max_val = 1.900000e+01 : f32} : (tensor<12x188x83x1xf32>) -> tensor<12x188x83x1xf32>
    return %2, %4 : tensor<49x28x1x96xi1>, tensor<12x188x83x1xf32>
  }
}
