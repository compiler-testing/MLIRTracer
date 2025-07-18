module {
  func.func @main(%arg0: tensor<39x8x11x89xf32>, %arg1: tensor<57x3x47x77xf32>, %arg2: tensor<57xf32>, %arg3: tensor<53x95x36x25xi64>, %arg4: tensor<1x1x36x25xi64>) -> (tensor<39x12x1x57xf32>, tensor<53x95x36x25xi64>, tensor<39x12x61x57xf32>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 39, 12, 61, 57>} : (tensor<39x8x11x89xf32>, tensor<57x3x47x77xf32>, tensor<57xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<39x12x61x57xf32>
    %1 = tosa.arithmetic_right_shift %arg3, %arg4 {round = false} : (tensor<53x95x36x25xi64>, tensor<1x1x36x25xi64>) -> tensor<53x95x36x25xi64>
    %2 = tosa.reduce_min %0 {axis = 2 : i32} : (tensor<39x12x61x57xf32>) -> tensor<39x12x1x57xf32>
    %3 = tosa.abs %2 : (tensor<39x12x1x57xf32>) -> tensor<39x12x1x57xf32>
    %4 = tosa.clz %1 : (tensor<53x95x36x25xi64>) -> tensor<53x95x36x25xi64>
    %5 = tosa.exp %0 : (tensor<39x12x61x57xf32>) -> tensor<39x12x61x57xf32>
    return %3, %4, %5 : tensor<39x12x1x57xf32>, tensor<53x95x36x25xi64>, tensor<39x12x61x57xf32>
  }
}
