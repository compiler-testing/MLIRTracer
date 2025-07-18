module {
  func.func @main(%arg0: tensor<16x79x8x83xf32>, %arg1: tensor<39x81x54x7xf32>, %arg2: tensor<39xf32>) -> (tensor<16x239x72x39xf32>, tensor<16x239x72x39xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 16, 239, 72, 39>} : (tensor<16x79x8x83xf32>, tensor<39x81x54x7xf32>, tensor<39xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<16x239x72x39xf32>
    %1 = tosa.greater_equal %0, %0 : (tensor<16x239x72x39xf32>, tensor<16x239x72x39xf32>) -> tensor<16x239x72x39xi1>
    %2 = tosa.floor %0 : (tensor<16x239x72x39xf32>) -> tensor<16x239x72x39xf32>
    %3 = tosa.reverse %1 {axis = 1 : i32} : (tensor<16x239x72x39xi1>) -> tensor<16x239x72x39xi1>
    %4 = tosa.bitwise_and %3, %3 : (tensor<16x239x72x39xi1>, tensor<16x239x72x39xi1>) -> tensor<16x239x72x39xi1>
    return %2, %4 : tensor<16x239x72x39xf32>, tensor<16x239x72x39xi1>
  }
}
