module {
  func.func @main(%arg0: tensor<73x51x38x97xf32>, %arg1: tensor<48x2x30x40xf32>, %arg2: tensor<48xf32>, %arg3: tensor<71xi1>) -> (tensor<1x12x216810x1xf32>, tensor<1xi1>, tensor<1xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 73, 55, 108, 48>} : (tensor<73x51x38x97xf32>, tensor<48x2x30x40xf32>, tensor<48xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<73x55x108x48xf32>
    %r_1 = tosa.const_shape {values = dense<[ 2, 8030, 648, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.reshape %0, %r_1 : (tensor<73x55x108x48xf32>, !tosa.shape<4>) -> tensor<2x8030x648x2xf32>
    %2 = tosa.clamp %1 {min_val = 3.000000e+01 : f32, max_val = 4.200000e+01 : f32} : (tensor<2x8030x648x2xf32>) -> tensor<2x8030x648x2xf32>
    %r_3 = tosa.const_shape {values = dense<[ 1, 12, 216810, 8 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.reshape %2, %r_3 : (tensor<2x8030x648x2xf32>, !tosa.shape<4>) -> tensor<1x12x216810x8xf32>
    %4 = tosa.reduce_sum %3 {axis = 3 : i32} : (tensor<1x12x216810x8xf32>) -> tensor<1x12x216810x1xf32>
    %5 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<71xi1>) -> tensor<1xi1>
    %6 = tosa.logical_not %5 : (tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.reduce_product %5 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %8 = tosa.sub %7, %7 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %4, %6, %8 : tensor<1x12x216810x1xf32>, tensor<1xi1>, tensor<1xi1>
  }
}
