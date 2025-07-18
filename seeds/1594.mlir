module {
  func.func @main(%arg0: tensor<58x17x64x27xf32>, %arg1: tensor<21x23x70x87xf32>, %arg2: tensor<21xf32>, %arg3: tensor<44x70x92x55xi32>, %arg4: tensor<44x70x1x55xi32>) -> (tensor<58x58x198x21xf32>, tensor<44x70x92x55xi32>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 58, 58, 198, 21>} : (tensor<58x17x64x27xf32>, tensor<21x23x70x87xf32>, tensor<21xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<58x58x198x21xf32>
    %1 = tosa.intdiv %arg3, %arg4 : (tensor<44x70x92x55xi32>, tensor<44x70x1x55xi32>) -> tensor<44x70x92x55xi32>
    return %0, %1 : tensor<58x58x198x21xf32>, tensor<44x70x92x55xi32>
  }
}
