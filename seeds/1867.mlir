module {
  func.func @main(%arg0: tensor<1x73x19xi1>, %arg1: tensor<1x19x4xi1>, %arg2: tensor<f32>) -> (tensor<1x1x1x1xi1>, tensor<i1>, tensor<2x73x4xi1>, tensor<1x73x4xi1>, tensor<1x73xi32>, tensor<f32>, tensor<f32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<1x73x19xi1>, tensor<1x19x4xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1x73x4xi1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<1x73x4xi1>, tensor<1x73x4xi1>) -> tensor<1x73x4xi1>
    %2 = tosa.exp %arg2 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.greater %2, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = tosa.logical_and %3, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %r_5 = tosa.const_shape {values = dense<[ 1, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.reshape %4, %r_5 : (tensor<i1>, !tosa.shape<4>) -> tensor<1x1x1x1xi1>
    %6 = tosa.rsqrt %2 : (tensor<f32>) -> tensor<f32>
    %7 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<1x73x4xi1>) -> tensor<1x73x4xi1>
    %8 = tosa.bitwise_or %4, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %9 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<1x73x4xi1>, tensor<1x73x4xi1>) -> tensor<2x73x4xi1>
    %10 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<1x73x4xi1>, tensor<1x73x4xi1>) -> tensor<1x73x4xi1>
    %11 = tosa.floor %2 : (tensor<f32>) -> tensor<f32>
    %12 = tosa.argmax %7 {axis = 2 : i32} : (tensor<1x73x4xi1>) -> tensor<1x73xi32>
    %13 = tosa.floor %6 : (tensor<f32>) -> tensor<f32>
    %14 = tosa.sigmoid %13 : (tensor<f32>) -> tensor<f32>
    %15 = tosa.sigmoid %11 : (tensor<f32>) -> tensor<f32>
    %16 = tosa.sigmoid %15 : (tensor<f32>) -> tensor<f32>
    %17 = tosa.sigmoid %16 : (tensor<f32>) -> tensor<f32>
    %18 = tosa.tanh %17 : (tensor<f32>) -> tensor<f32>
    %19 = tosa.abs %18 : (tensor<f32>) -> tensor<f32>
    %20 = tosa.floor %14 : (tensor<f32>) -> tensor<f32>
    return %5, %8, %9, %10, %12, %19, %20 : tensor<1x1x1x1xi1>, tensor<i1>, tensor<2x73x4xi1>, tensor<1x73x4xi1>, tensor<1x73xi32>, tensor<f32>, tensor<f32>
  }
}
