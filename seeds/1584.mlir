module {
  func.func @main(%arg0: tensor<17x56x69xi64>, %arg1: tensor<17x69x50xi64>, %arg2: tensor<82x83x77x50xf32>, %arg3: tensor<18x93x98x71xf32>, %arg4: tensor<18xf32>, %arg5: tensor<87x91x8x21xf32>, %arg6: tensor<87xf32>) -> (tensor<17x56x50xi64>, tensor<82x615x187x87xf32>, tensor<82x261x177x36xf32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<17x56x69xi64>, tensor<17x69x50xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<17x56x50xi64>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 82, 261, 177, 18>} : (tensor<82x83x77x50xf32>, tensor<18x93x98x71xf32>, tensor<18xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<82x261x177x18xf32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %1, %arg5, %arg6, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 82, 615, 187, 87>} : (tensor<82x261x177x18xf32>, tensor<87x91x8x21xf32>, tensor<87xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<82x615x187x87xf32>
    %3 = tosa.concat %1, %1 {axis = 3 : i32} : (tensor<82x261x177x18xf32>, tensor<82x261x177x18xf32>) -> tensor<82x261x177x36xf32>
    %4 = tosa.exp %3 : (tensor<82x261x177x36xf32>) -> tensor<82x261x177x36xf32>
    return %0, %2, %4 : tensor<17x56x50xi64>, tensor<82x615x187x87xf32>, tensor<82x261x177x36xf32>
  }
}
