module {
  func.func @main(%arg0: tensor<90x60x35x25xf32>, %arg1: tensor<96x64x64x74xf32>, %arg2: tensor<96xf32>) -> (tensor<180x127x1x96xi1>, tensor<90x127x101x96xf32>, tensor<90x127x101x96xf32>, tensor<90x127x101x96xf32>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 90, 127, 101, 96>} : (tensor<90x60x35x25xf32>, tensor<96x64x64x74xf32>, tensor<96xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<90x127x101x96xf32>
    %1 = tosa.abs %0 : (tensor<90x127x101x96xf32>) -> tensor<90x127x101x96xf32>
    %2 = tosa.log %1 : (tensor<90x127x101x96xf32>) -> tensor<90x127x101x96xf32>
    %3 = tosa.greater_equal %2, %2 : (tensor<90x127x101x96xf32>, tensor<90x127x101x96xf32>) -> tensor<90x127x101x96xi1>
    %4 = tosa.pow %0, %1 : (tensor<90x127x101x96xf32>, tensor<90x127x101x96xf32>) -> tensor<90x127x101x96xf32>
    %5 = tosa.ceil %4 : (tensor<90x127x101x96xf32>) -> tensor<90x127x101x96xf32>
    %6 = tosa.exp %4 : (tensor<90x127x101x96xf32>) -> tensor<90x127x101x96xf32>
    %7 = tosa.concat %3, %3 {axis = 0 : i32} : (tensor<90x127x101x96xi1>, tensor<90x127x101x96xi1>) -> tensor<180x127x101x96xi1>
    %8 = tosa.reduce_any %7 {axis = 2 : i32} : (tensor<180x127x101x96xi1>) -> tensor<180x127x1x96xi1>
    %9 = tosa.minimum %6, %0 : (tensor<90x127x101x96xf32>, tensor<90x127x101x96xf32>) -> tensor<90x127x101x96xf32>
    %10 = tosa.sigmoid %4 : (tensor<90x127x101x96xf32>) -> tensor<90x127x101x96xf32>
    %11 = tosa.pow %5, %2 : (tensor<90x127x101x96xf32>, tensor<90x127x101x96xf32>) -> tensor<90x127x101x96xf32>
    return %8, %9, %10, %11 : tensor<180x127x1x96xi1>, tensor<90x127x101x96xf32>, tensor<90x127x101x96xf32>, tensor<90x127x101x96xf32>
  }
}
