module {
  func.func @main(%arg0: tensor<1x29x36x32xi32>, %arg1: tensor<18x14x90x71xi1>, %arg2: tensor<59x89x53xf32>) -> (tensor<3x38340x1xi1>, tensor<18x1x90x71xi1>, tensor<59x89x53xf32>, tensor<1x1x90x71xi1>, tensor<59x89x53xi1>, tensor<1x1x36x32xi32>, tensor<59x89x53xf32>, tensor<59x89x53xi1>, tensor<59x89x53xf32>) {
    %0 = tosa.reverse %arg0 {axis = 2 : i32} : (tensor<1x29x36x32xi32>) -> tensor<1x29x36x32xi32>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<1x29x36x32xi32>) -> tensor<1x1x36x32xi32>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<1x1x36x32xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x36x32xi32>
    %3 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<18x14x90x71xi1>) -> tensor<18x1x90x71xi1>
    %4 = tosa.logical_not %3 : (tensor<18x1x90x71xi1>) -> tensor<18x1x90x71xi1>
    %5 = tosa.exp %arg2 : (tensor<59x89x53xf32>) -> tensor<59x89x53xf32>
    %r_6 = tosa.const_shape {values = dense<[ 3, 38340, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.reshape %4, %r_6 : (tensor<18x1x90x71xi1>, !tosa.shape<3>) -> tensor<3x38340x1xi1>
    %7 = tosa.intdiv %1, %2 : (tensor<1x1x36x32xi32>, tensor<1x1x36x32xi32>) -> tensor<1x1x36x32xi32>
    %8 = tosa.logical_left_shift %3, %4 : (tensor<18x1x90x71xi1>, tensor<18x1x90x71xi1>) -> tensor<18x1x90x71xi1>
    %9 = tosa.floor %5 : (tensor<59x89x53xf32>) -> tensor<59x89x53xf32>
    %10 = tosa.sigmoid %5 : (tensor<59x89x53xf32>) -> tensor<59x89x53xf32>
    %11 = tosa.ceil %9 : (tensor<59x89x53xf32>) -> tensor<59x89x53xf32>
    %12 = tosa.reduce_all %3 {axis = 0 : i32} : (tensor<18x1x90x71xi1>) -> tensor<1x1x90x71xi1>
    %13 = tosa.greater_equal %10, %5 : (tensor<59x89x53xf32>, tensor<59x89x53xf32>) -> tensor<59x89x53xi1>
    %14 = tosa.logical_right_shift %13, %13 : (tensor<59x89x53xi1>, tensor<59x89x53xi1>) -> tensor<59x89x53xi1>
    %15 = tosa.intdiv %7, %2 : (tensor<1x1x36x32xi32>, tensor<1x1x36x32xi32>) -> tensor<1x1x36x32xi32>
    %16 = tosa.greater_equal %10, %5 : (tensor<59x89x53xf32>, tensor<59x89x53xf32>) -> tensor<59x89x53xi1>
    %17 = tosa.tanh %10 : (tensor<59x89x53xf32>) -> tensor<59x89x53xf32>
    %18 = tosa.clamp %17 {min_val = -4.400000e+01 : f32, max_val = 1.800000e+01 : f32} : (tensor<59x89x53xf32>) -> tensor<59x89x53xf32>
    %19 = "tosa.const"() {values = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
    %20 = tosa.transpose %18 {perms = array<i32: 0, 1, 2>} : (tensor<59x89x53xf32>) -> tensor<59x89x53xf32>
    %21 = tosa.sigmoid %10 : (tensor<59x89x53xf32>) -> tensor<59x89x53xf32>
    %22 = tosa.logical_and %13, %16 : (tensor<59x89x53xi1>, tensor<59x89x53xi1>) -> tensor<59x89x53xi1>
    %23 = tosa.floor %21 : (tensor<59x89x53xf32>) -> tensor<59x89x53xf32>
    %r_24 = tosa.const_shape {values = dense<[ 59, 89, 53 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %24 = tosa.reshape %23, %r_24 : (tensor<59x89x53xf32>, !tosa.shape<3>) -> tensor<59x89x53xf32>
    return %6, %8, %11, %12, %14, %15, %20, %22, %24 : tensor<3x38340x1xi1>, tensor<18x1x90x71xi1>, tensor<59x89x53xf32>, tensor<1x1x90x71xi1>, tensor<59x89x53xi1>, tensor<1x1x36x32xi32>, tensor<59x89x53xf32>, tensor<59x89x53xi1>, tensor<59x89x53xf32>
  }
}
