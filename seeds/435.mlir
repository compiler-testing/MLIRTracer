module {
  func.func @main(%arg0: tensor<17x64x28x34xf32>, %arg1: tensor<93x58x16x86xf32>, %arg2: tensor<93xf32>) -> tensor<17x124x74x93xf32> {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 17, 124, 74, 93>} : (tensor<17x64x28x34xf32>, tensor<93x58x16x86xf32>, tensor<93xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<17x124x74x93xf32>
    %1 = tosa.pow %0, %0 : (tensor<17x124x74x93xf32>, tensor<17x124x74x93xf32>) -> tensor<17x124x74x93xf32>
    return %1 : tensor<17x124x74x93xf32>
  }
}
