module {
  func.func @main(%arg0: tensor<6x76x28x81x34x94xf32>, %arg1: tensor<45x8x89x78xf32>, %arg2: tensor<95x40x39x54xf32>, %arg3: tensor<95xf32>) -> (tensor<34x81x76x28x6x94xf32>, tensor<45x50x1x95xf32>) {
    %0 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<6x76x28x81x34x94xf32>) -> tensor<34x81x76x28x6x94xf32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 45, 50, 131, 95>} : (tensor<45x8x89x78xf32>, tensor<95x40x39x54xf32>, tensor<95xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<45x50x131x95xf32>
    %3 = tosa.reduce_sum %2 {axis = 2 : i32} : (tensor<45x50x131x95xf32>) -> tensor<45x50x1x95xf32>
    return %1, %3 : tensor<34x81x76x28x6x94xf32>, tensor<45x50x1x95xf32>
  }
}
