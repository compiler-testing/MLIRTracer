module {
  func.func @main(%arg0: tensor<41x47x51x11x59x76xi32>, %arg1: tensor<6x93x47x25xf32>, %arg2: tensor<90x41x27x52xf32>, %arg3: tensor<90xf32>, %arg4: tensor<5xi1>, %arg5: tensor<5xi1>) -> (tensor<41x47x51x11x59x76xi32>, tensor<1x229x123x1xf32>, tensor<5xi1>, tensor<6x229x123x90xf32>) {
    %0 = tosa.identity %arg0 : (tensor<41x47x51x11x59x76xi32>) -> tensor<41x47x51x11x59x76xi32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 6, 229, 123, 90>} : (tensor<6x93x47x25xf32>, tensor<90x41x27x52xf32>, tensor<90xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<6x229x123x90xf32>
    %2 = tosa.bitwise_not %0 : (tensor<41x47x51x11x59x76xi32>) -> tensor<41x47x51x11x59x76xi32>
    %3 = tosa.reduce_sum %1 {axis = 3 : i32} : (tensor<6x229x123x90xf32>) -> tensor<6x229x123x1xf32>
    %4 = tosa.reduce_max %3 {axis = 3 : i32} : (tensor<6x229x123x1xf32>) -> tensor<6x229x123x1xf32>
    %5 = tosa.logical_xor %arg4, %arg5 : (tensor<5xi1>, tensor<5xi1>) -> tensor<5xi1>
    %6 = tosa.reduce_sum %4 {axis = 0 : i32} : (tensor<6x229x123x1xf32>) -> tensor<1x229x123x1xf32>
    %7 = tosa.logical_not %5 : (tensor<5xi1>) -> tensor<5xi1>
    %8 = tosa.sigmoid %1 : (tensor<6x229x123x90xf32>) -> tensor<6x229x123x90xf32>
    return %2, %6, %7, %8 : tensor<41x47x51x11x59x76xi32>, tensor<1x229x123x1xf32>, tensor<5xi1>, tensor<6x229x123x90xf32>
  }
}
