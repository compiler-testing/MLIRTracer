module {
  func.func @main(%arg0: tensor<100x55xi1>, %arg1: tensor<42xi32>, %arg2: tensor<42xi32>, %arg3: tensor<59x70x98x11xf32>, %arg4: tensor<72x84x43x10xf32>, %arg5: tensor<72xf32>) -> (tensor<3x55xi1>, tensor<42xi32>, tensor<59x156x240x72xf32>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<100x55xi1>) -> tensor<1x55xi1>
    %t_1 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %0, %t_1 : (tensor<1x55xi1>, !tosa.shape<2>) -> tensor<3x55xi1>
    %2 = tosa.intdiv %arg1, %arg2 : (tensor<42xi32>, tensor<42xi32>) -> tensor<42xi32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 1>, stride = array<i64: 1, 2>, out_shape = array<i64: 59, 156, 240, 72>} : (tensor<59x70x98x11xf32>, tensor<72x84x43x10xf32>, tensor<72xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<59x156x240x72xf32>
    return %1, %2, %3 : tensor<3x55xi1>, tensor<42xi32>, tensor<59x156x240x72xf32>
  }
}
