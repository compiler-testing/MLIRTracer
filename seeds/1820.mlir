module {
  func.func @main(%arg0: tensor<13xi1>, %arg1: tensor<19x34x54x15xf32>, %arg2: tensor<96x87x83x70xf32>, %arg3: tensor<96xf32>) -> (tensor<1xi1>, tensor<19x124x138x96xf32>, tensor<19x124x138x96xf32>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<13xi1>) -> tensor<1xi1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 19, 124, 138, 96>} : (tensor<19x34x54x15xf32>, tensor<96x87x83x70xf32>, tensor<96xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<19x124x138x96xf32>
    %4 = tosa.exp %3 : (tensor<19x124x138x96xf32>) -> tensor<19x124x138x96xf32>
    %5 = tosa.tanh %3 : (tensor<19x124x138x96xf32>) -> tensor<19x124x138x96xf32>
    return %2, %4, %5 : tensor<1xi1>, tensor<19x124x138x96xf32>, tensor<19x124x138x96xf32>
  }
}
