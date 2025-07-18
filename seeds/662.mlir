module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<84x96x11x76xf32>, %arg2: tensor<43x21x27x30xf32>, %arg3: tensor<43xf32>, %arg4: tensor<8x34xi64>, %arg5: tensor<8x34xi64>, %arg6: tensor<66x68x52xi1>) -> (tensor<f32>, tensor<8x34xi64>, tensor<84x50x43xi32>, tensor<66x1x52xi1>, tensor<168x357x200x43xf32>, tensor<168x357x200x43xf32>, tensor<168x357x200x43xf32>, tensor<84x1x50x86xf32>) {
    %0 = tosa.floor %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.floor %0 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.identity %1 : (tensor<f32>) -> tensor<f32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 1>, stride = array<i64: 1, 2>, out_shape = array<i64: 84, 119, 50, 43>} : (tensor<84x96x11x76xf32>, tensor<43x21x27x30xf32>, tensor<43xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<84x119x50x43xf32>
    %4 = tosa.minimum %3, %3 : (tensor<84x119x50x43xf32>, tensor<84x119x50x43xf32>) -> tensor<84x119x50x43xf32>
    %5 = tosa.sub %4, %4 : (tensor<84x119x50x43xf32>, tensor<84x119x50x43xf32>) -> tensor<84x119x50x43xf32>
    %6 = tosa.bitwise_xor %arg4, %arg5 : (tensor<8x34xi64>, tensor<8x34xi64>) -> tensor<8x34xi64>
    %7 = tosa.argmax %3 {axis = 1 : i32} : (tensor<84x119x50x43xf32>) -> tensor<84x50x43xi32>
    %8 = tosa.reduce_all %arg6 {axis = 1 : i32} : (tensor<66x68x52xi1>) -> tensor<66x1x52xi1>
    %9 = tosa.concat %3, %3 {axis = 2 : i32} : (tensor<84x119x50x43xf32>, tensor<84x119x50x43xf32>) -> tensor<84x119x100x43xf32>
    %t_10 = tosa.const_shape {values = dense<[ 2, 3, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.tile %9, %t_10 : (tensor<84x119x100x43xf32>, !tosa.shape<4>) -> tensor<168x357x200x43xf32>
    %11 = tosa.pow %10, %10 : (tensor<168x357x200x43xf32>, tensor<168x357x200x43xf32>) -> tensor<168x357x200x43xf32>
    %12 = tosa.sub %10, %10 : (tensor<168x357x200x43xf32>, tensor<168x357x200x43xf32>) -> tensor<168x357x200x43xf32>
    %13 = tosa.concat %5, %4 {axis = 3 : i32} : (tensor<84x119x50x43xf32>, tensor<84x119x50x43xf32>) -> tensor<84x119x50x86xf32>
    %14 = tosa.tanh %10 : (tensor<168x357x200x43xf32>) -> tensor<168x357x200x43xf32>
    %15 = tosa.reduce_max %13 {axis = 1 : i32} : (tensor<84x119x50x86xf32>) -> tensor<84x1x50x86xf32>
    return %2, %6, %7, %8, %11, %12, %14, %15 : tensor<f32>, tensor<8x34xi64>, tensor<84x50x43xi32>, tensor<66x1x52xi1>, tensor<168x357x200x43xf32>, tensor<168x357x200x43xf32>, tensor<168x357x200x43xf32>, tensor<84x1x50x86xf32>
  }
}
