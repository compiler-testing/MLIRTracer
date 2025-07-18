module {
  func.func @main(%arg0: tensor<31x19x18x97x28xf32>, %arg1: tensor<71xi1>, %arg2: tensor<44x11x78x34xf32>, %arg3: tensor<15x17x20x15xf32>, %arg4: tensor<15xf32>) -> (tensor<31x19x18x97x28xf32>, tensor<44x31x177x15xf32>, tensor<1xi1>) {
    %0 = tosa.exp %arg0 : (tensor<31x19x18x97x28xf32>) -> tensor<31x19x18x97x28xf32>
    %1 = tosa.reciprocal %0 : (tensor<31x19x18x97x28xf32>) -> tensor<31x19x18x97x28xf32>
    %2 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<71xi1>) -> tensor<1xi1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 44, 31, 177, 15>} : (tensor<44x11x78x34xf32>, tensor<15x17x20x15xf32>, tensor<15xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<44x31x177x15xf32>
    %4 = tosa.reduce_any %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %1, %3, %4 : tensor<31x19x18x97x28xf32>, tensor<44x31x177x15xf32>, tensor<1xi1>
  }
}
