module {
  func.func @main(%arg0: tensor<73x56x65xi8>, %arg1: tensor<73x65x14xi8>, %arg2: tensor<f32>) -> (tensor<2x1xi32>, tensor<73x56xi32>, tensor<f32>, tensor<f32>, tensor<73x56x14xi8>, tensor<1x56x14xi8>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<73x56x65xi8>, tensor<73x65x14xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<73x56x14xi8>
    %1 = tosa.clamp %0 {min_val = -63 : i8, max_val = 11 : i8} : (tensor<73x56x14xi8>) -> tensor<73x56x14xi8>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<73x56x14xi8>) -> tensor<1x56x14xi8>
    %3 = tosa.argmax %2 {axis = 2 : i32} : (tensor<1x56x14xi8>) -> tensor<1x56xi32>
    %4 = tosa.reduce_max %3 {axis = 0 : i32} : (tensor<1x56xi32>) -> tensor<1x56xi32>
    %5 = tosa.reduce_sum %4 {axis = 1 : i32} : (tensor<1x56xi32>) -> tensor<1x1xi32>
    %6 = tosa.exp %arg2 : (tensor<f32>) -> tensor<f32>
    %7 = tosa.concat %5, %5 {axis = 0 : i32} : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<2x1xi32>
    %8 = tosa.log %6 : (tensor<f32>) -> tensor<f32>
    %9 = tosa.sigmoid %6 : (tensor<f32>) -> tensor<f32>
    %10 = tosa.argmax %1 {axis = 2 : i32} : (tensor<73x56x14xi8>) -> tensor<73x56xi32>
    %11 = tosa.rsqrt %8 : (tensor<f32>) -> tensor<f32>
    %12 = tosa.add %9, %11 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %13 = tosa.pow %11, %11 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %14 = tosa.bitwise_or %1, %0 : (tensor<73x56x14xi8>, tensor<73x56x14xi8>) -> tensor<73x56x14xi8>
    %15 = tosa.logical_right_shift %2, %2 : (tensor<1x56x14xi8>, tensor<1x56x14xi8>) -> tensor<1x56x14xi8>
    return %7, %10, %12, %13, %14, %15 : tensor<2x1xi32>, tensor<73x56xi32>, tensor<f32>, tensor<f32>, tensor<73x56x14xi8>, tensor<1x56x14xi8>
  }
}
