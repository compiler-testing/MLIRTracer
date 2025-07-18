module {
  func.func @main(%arg0: tensor<4x65x89xi1>, %arg1: tensor<60x30xi32>, %arg2: tensor<60x1xi32>, %arg3: tensor<18x55x82x40xf32>, %arg4: tensor<4x61x75x75xf32>, %arg5: tensor<4xf32>) -> (tensor<4x65x89xi1>, tensor<18x119x158x1xf32>, tensor<18x119x158x4xf32>, tensor<60x30xi32>, tensor<18x4x119x158xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<4x65x89xi1>) -> tensor<4x65x89xi1>
    %1 = tosa.minimum %arg1, %arg2 : (tensor<60x30xi32>, tensor<60x1xi32>) -> tensor<60x30xi32>
    %2 = tosa.identity %1 : (tensor<60x30xi32>) -> tensor<60x30xi32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 18, 119, 158, 4>} : (tensor<18x55x82x40xf32>, tensor<4x61x75x75xf32>, tensor<4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<18x119x158x4xf32>
    %4 = tosa.rsqrt %3 : (tensor<18x119x158x4xf32>) -> tensor<18x119x158x4xf32>
    %5 = tosa.pow %3, %3 : (tensor<18x119x158x4xf32>, tensor<18x119x158x4xf32>) -> tensor<18x119x158x4xf32>
    %6 = tosa.reduce_max %5 {axis = 3 : i32} : (tensor<18x119x158x4xf32>) -> tensor<18x119x158x1xf32>
    %7 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %8 = tosa.transpose %5 {perms = array<i32: 0, 3, 1, 2>} : (tensor<18x119x158x4xf32>) -> tensor<18x4x119x158xf32>
    %9 = tosa.clamp %4 {min_val = -4.500000e+01 : f32, max_val = 1.000000e+00 : f32} : (tensor<18x119x158x4xf32>) -> tensor<18x119x158x4xf32>
    %10 = tosa.intdiv %2, %1 : (tensor<60x30xi32>, tensor<60x30xi32>) -> tensor<60x30xi32>
    %in_zp_11 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_11 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %11 = tosa.negate %8, %in_zp_11, %out_zp_11 : (tensor<18x4x119x158xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<18x4x119x158xf32>
    return %0, %6, %9, %10, %11 : tensor<4x65x89xi1>, tensor<18x119x158x1xf32>, tensor<18x119x158x4xf32>, tensor<60x30xi32>, tensor<18x4x119x158xf32>
  }
}
