module {
  func.func @main(%arg0: tensor<56x86x62x22xf32>, %arg1: tensor<38x44x42x4xf32>, %arg2: tensor<38xf32>, %arg3: tensor<78x25x20x54xi1>) -> (tensor<56x132x106x38xf32>, tensor<1x25x20x1xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<56x86x62x22xf32>) -> tensor<56x86x62x22xf32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %0, %arg1, %arg2, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 56, 132, 106, 38>} : (tensor<56x86x62x22xf32>, tensor<38x44x42x4xf32>, tensor<38xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<56x132x106x38xf32>
    %2 = tosa.ceil %1 : (tensor<56x132x106x38xf32>) -> tensor<56x132x106x38xf32>
    %3 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<78x25x20x54xi1>) -> tensor<1x25x20x54xi1>
    %4 = tosa.reduce_any %3 {axis = 3 : i32} : (tensor<1x25x20x54xi1>) -> tensor<1x25x20x1xi1>
    %5 = tosa.sub %4, %4 : (tensor<1x25x20x1xi1>, tensor<1x25x20x1xi1>) -> tensor<1x25x20x1xi1>
    return %2, %5 : tensor<56x132x106x38xf32>, tensor<1x25x20x1xi1>
  }
}
