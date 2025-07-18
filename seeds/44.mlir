module {
  func.func @main(%arg0: tensor<32xi32>, %arg1: tensor<1xi32>, %arg2: tensor<87x23x60x92xf32>, %arg3: tensor<54x18x3x79xf32>, %arg4: tensor<54xf32>) -> (tensor<87x42x1x54xf32>, tensor<32xi32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<32xi32>, tensor<1xi32>) -> tensor<32xi32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 87, 42, 124, 54>} : (tensor<87x23x60x92xf32>, tensor<54x18x3x79xf32>, tensor<54xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<87x42x124x54xf32>
    %2 = tosa.ceil %1 : (tensor<87x42x124x54xf32>) -> tensor<87x42x124x54xf32>
    %3 = tosa.reduce_product %2 {axis = 2 : i32} : (tensor<87x42x124x54xf32>) -> tensor<87x42x1x54xf32>
    %4 = tosa.logical_left_shift %0, %0 : (tensor<32xi32>, tensor<32xi32>) -> tensor<32xi32>
    return %3, %4 : tensor<87x42x1x54xf32>, tensor<32xi32>
  }
}
