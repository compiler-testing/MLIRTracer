module {
  func.func @main(%arg0: tensor<44xi8>, %arg1: tensor<100x88x35xi1>, %arg2: tensor<74x7x65x50xf32>, %arg3: tensor<26x68x56x20xf32>, %arg4: tensor<26xf32>) -> (tensor<i32>, tensor<74x84x123x26xf32>, tensor<100x1x1xi1>, tensor<74x84x123x26xi1>, tensor<74x84x123x26xf32>, tensor<74x84x123x26xf32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<44xi8>) -> tensor<1xi8>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<1xi8>) -> tensor<i32>
    %2 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<100x88x35xi1>) -> tensor<100x1x35xi1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 74, 84, 123, 26>} : (tensor<74x7x65x50xf32>, tensor<26x68x56x20xf32>, tensor<26xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<74x84x123x26xf32>
    %4 = tosa.floor %3 : (tensor<74x84x123x26xf32>) -> tensor<74x84x123x26xf32>
    %5 = tosa.logical_or %2, %2 : (tensor<100x1x35xi1>, tensor<100x1x35xi1>) -> tensor<100x1x35xi1>
    %6 = tosa.reduce_sum %5 {axis = 2 : i32} : (tensor<100x1x35xi1>) -> tensor<100x1x1xi1>
    %7 = tosa.logical_left_shift %6, %6 : (tensor<100x1x1xi1>, tensor<100x1x1xi1>) -> tensor<100x1x1xi1>
    %8 = tosa.greater_equal %3, %3 : (tensor<74x84x123x26xf32>, tensor<74x84x123x26xf32>) -> tensor<74x84x123x26xi1>
    %9 = tosa.pow %3, %3 : (tensor<74x84x123x26xf32>, tensor<74x84x123x26xf32>) -> tensor<74x84x123x26xf32>
    %10 = tosa.log %3 : (tensor<74x84x123x26xf32>) -> tensor<74x84x123x26xf32>
    return %1, %4, %7, %8, %9, %10 : tensor<i32>, tensor<74x84x123x26xf32>, tensor<100x1x1xi1>, tensor<74x84x123x26xi1>, tensor<74x84x123x26xf32>, tensor<74x84x123x26xf32>
  }
}
