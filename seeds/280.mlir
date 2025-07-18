module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<21x45x72x17x51xi1>, %arg2: tensor<21x1x72x17x51xi1>) -> (tensor<f32>, tensor<21x45x72x17x51xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<f32>, tensor<1xf32>, tensor<1xf32>) -> tensor<f32>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<21x45x72x17x51xi1>, tensor<21x1x72x17x51xi1>) -> tensor<21x45x72x17x51xi1>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<21x45x72x17x51xi1>, tensor<21x45x72x17x51xi1>) -> tensor<21x45x72x17x51xi1>
    return %0, %2 : tensor<f32>, tensor<21x45x72x17x51xi1>
  }
}
