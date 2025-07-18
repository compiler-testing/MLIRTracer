module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<13x76x43x5xf32>, %arg3: tensor<87x22x65x76xf32>, %arg4: tensor<87xf32>) -> (tensor<13x176x153x87xf32>, tensor<i1>, tensor<13x176x153x87xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 13, 176, 153, 87>} : (tensor<13x76x43x5xf32>, tensor<87x22x65x76xf32>, tensor<87xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<13x176x153x87xf32>
    %2 = tosa.logical_left_shift %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.log %1 : (tensor<13x176x153x87xf32>) -> tensor<13x176x153x87xf32>
    %4 = tosa.logical_right_shift %0, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.floor %1 : (tensor<13x176x153x87xf32>) -> tensor<13x176x153x87xf32>
    %6 = tosa.exp %5 : (tensor<13x176x153x87xf32>) -> tensor<13x176x153x87xf32>
    return %3, %4, %6 : tensor<13x176x153x87xf32>, tensor<i1>, tensor<13x176x153x87xf32>
  }
}
