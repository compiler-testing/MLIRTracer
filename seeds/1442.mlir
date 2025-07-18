module {
  func.func @main(%arg0: tensor<41x23xi32>, %arg1: tensor<41x23xi32>, %arg2: tensor<68x43x67x2xf32>, %arg3: tensor<81x46x83x30xf32>, %arg4: tensor<81xf32>) -> (tensor<68x1x217x81xf32>, tensor<1x23xi32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<41x23xi32>, tensor<41x23xi32>) -> tensor<41x23xi32>
    %1 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<41x23xi32>) -> tensor<1x23xi32>
    %2 = tosa.clamp %1 {min_val = -11 : i32, max_val = -7 : i32} : (tensor<1x23xi32>) -> tensor<1x23xi32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 68, 133, 217, 81>} : (tensor<68x43x67x2xf32>, tensor<81x46x83x30xf32>, tensor<81xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<68x133x217x81xf32>
    %4 = tosa.reduce_max %3 {axis = 1 : i32} : (tensor<68x133x217x81xf32>) -> tensor<68x1x217x81xf32>
    %5 = tosa.clamp %2 {min_val = -11 : i32, max_val = -7 : i32} : (tensor<1x23xi32>) -> tensor<1x23xi32>
    return %4, %5 : tensor<68x1x217x81xf32>, tensor<1x23xi32>
  }
}
