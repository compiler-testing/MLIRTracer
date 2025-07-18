module {
  func.func @main(%arg0: tensor<31x69x65xi8>, %arg1: tensor<29xf32>, %arg2: tensor<1xf32>) -> (tensor<1x69x65xi8>, tensor<29xf32>, tensor<i1>, tensor<1xi1>, tensor<1x1x1xi1>, tensor<29xf32>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<31x69x65xi8>) -> tensor<1x69x65xi8>
    %1 = tosa.bitwise_and %0, %0 : (tensor<1x69x65xi8>, tensor<1x69x65xi8>) -> tensor<1x69x65xi8>
    %2 = tosa.pow %arg1, %arg2 : (tensor<29xf32>, tensor<1xf32>) -> tensor<29xf32>
    %3 = tosa.reciprocal %2 : (tensor<29xf32>) -> tensor<29xf32>
    %4 = tosa.concat %2, %3 {axis = 0 : i32} : (tensor<29xf32>, tensor<29xf32>) -> tensor<58xf32>
    %5 = tosa.clamp %4 {min_val = -3.300000e+01 : f32, max_val = 4.500000e+01 : f32} : (tensor<58xf32>) -> tensor<58xf32>
    %6 = tosa.add %3, %3 : (tensor<29xf32>, tensor<29xf32>) -> tensor<29xf32>
    %7 = tosa.equal %5, %5 : (tensor<58xf32>, tensor<58xf32>) -> tensor<58xi1>
    %8 = tosa.reverse %7 {axis = 0 : i32} : (tensor<58xi1>) -> tensor<58xi1>
    %in_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %9 = tosa.negate %8, %in_zp_9, %out_zp_9 : (tensor<58xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<58xi1>
    %10 = tosa.sub %6, %3 : (tensor<29xf32>, tensor<29xf32>) -> tensor<29xf32>
    %11 = tosa.reduce_max %9 {axis = 0 : i32} : (tensor<58xi1>) -> tensor<1xi1>
    %12 = tosa.log %10 : (tensor<29xf32>) -> tensor<29xf32>
    %13 = tosa.reduce_all %11 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %14 = tosa.logical_not %13 : (tensor<1xi1>) -> tensor<1xi1>
    %15 = tosa.reduce_all %11 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %16 = tosa.argmax %14 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %17 = tosa.greater_equal %16, %16 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %18 = tosa.arithmetic_right_shift %17, %17 {round = false} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %19 = tosa.reduce_any %15 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %20 = tosa.reduce_sum %15 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %r_21 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %21 = tosa.reshape %19, %r_21 : (tensor<1xi1>, !tosa.shape<3>) -> tensor<1x1x1xi1>
    %22 = tosa.minimum %3, %10 : (tensor<29xf32>, tensor<29xf32>) -> tensor<29xf32>
    return %1, %12, %18, %20, %21, %22 : tensor<1x69x65xi8>, tensor<29xf32>, tensor<i1>, tensor<1xi1>, tensor<1x1x1xi1>, tensor<29xf32>
  }
}
