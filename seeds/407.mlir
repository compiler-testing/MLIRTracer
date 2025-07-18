module {
  func.func @main(%arg0: tensor<19x4x21x9x28xi64>, %arg1: tensor<67x15xi8>, %arg2: tensor<64x14x76xf32>, %arg3: tensor<97x5x4x81xf32>, %arg4: tensor<85x73x18x65xf32>, %arg5: tensor<85xf32>) -> (tensor<19x4x21x9x28xi64>, tensor<67x15xi8>, tensor<64x14x76xf32>, tensor<97x85x28x85xf32>) {
    %0 = tosa.identity %arg0 : (tensor<19x4x21x9x28xi64>) -> tensor<19x4x21x9x28xi64>
    %1 = tosa.reverse %arg1 {axis = 1 : i32} : (tensor<67x15xi8>) -> tensor<67x15xi8>
    %2 = tosa.sigmoid %arg2 : (tensor<64x14x76xf32>) -> tensor<64x14x76xf32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 97, 85, 28, 85>} : (tensor<97x5x4x81xf32>, tensor<85x73x18x65xf32>, tensor<85xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<97x85x28x85xf32>
    %4 = tosa.log %3 : (tensor<97x85x28x85xf32>) -> tensor<97x85x28x85xf32>
    return %0, %1, %2, %4 : tensor<19x4x21x9x28xi64>, tensor<67x15xi8>, tensor<64x14x76xf32>, tensor<97x85x28x85xf32>
  }
}
