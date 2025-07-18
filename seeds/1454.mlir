module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<f32>, %arg3: tensor<29xi1>, %arg4: tensor<38x5xi8>, %arg5: tensor<38x1xi8>) -> (tensor<i8>, tensor<f32>, tensor<f32>, tensor<1xi1>, tensor<1x5xi8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %1 = tosa.bitwise_not %0 : (tensor<i8>) -> tensor<i8>
    %2 = tosa.rsqrt %arg2 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<29xi1>) -> tensor<1xi1>
    %4 = tosa.reduce_all %3 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.logical_right_shift %4, %3 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.abs %5 : (tensor<1xi1>) -> tensor<1xi1>
    %in_zp_7 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_7 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %7 = tosa.negate %2, %in_zp_7, %out_zp_7 : (tensor<f32>, tensor<1xf32>, tensor<1xf32>) -> tensor<f32>
    %in_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %8 = tosa.negate %2, %in_zp_8, %out_zp_8 : (tensor<f32>, tensor<1xf32>, tensor<1xf32>) -> tensor<f32>
    %9 = tosa.minimum %arg4, %arg5 : (tensor<38x5xi8>, tensor<38x1xi8>) -> tensor<38x5xi8>
    %10 = tosa.ceil %2 : (tensor<f32>) -> tensor<f32>
    %r_11 = tosa.const_shape {values = dense<[ 1, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %11 = tosa.reshape %8, %r_11 : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
    %12 = tosa.bitwise_and %6, %3 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %13 = tosa.logical_not %12 : (tensor<1xi1>) -> tensor<1xi1>
    %14 = tosa.reduce_sum %9 {axis = 0 : i32} : (tensor<38x5xi8>) -> tensor<1x5xi8>
    %15 = tosa.pow %11, %11 : (tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %16 = tosa.reduce_product %11 {axis = 0 : i32} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    return %1, %7, %10, %13, %14, %15, %16 : tensor<i8>, tensor<f32>, tensor<f32>, tensor<1xi1>, tensor<1x5xi8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>
  }
}
