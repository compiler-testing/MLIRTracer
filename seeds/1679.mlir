module {
  func.func @main(%arg0: tensor<26x77x99x56xi16>, %arg1: tensor<90x9x75xi1>, %arg2: tensor<84x7x77x9xf32>) -> (tensor<77x99x56xi32>, tensor<84x7x77x9xf32>, tensor<1x1xi32>, tensor<1x1x1xi1>, tensor<1x1xi32>, tensor<84x7x77x9xf32>, tensor<1x9x1xi1>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<26x77x99x56xi16>) -> tensor<77x99x56xi32>
    %1 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<90x9x75xi1>) -> tensor<1x9x75xi1>
    %2 = tosa.reduce_any %1 {axis = 2 : i32} : (tensor<1x9x75xi1>) -> tensor<1x9x1xi1>
    %3 = tosa.ceil %arg2 : (tensor<84x7x77x9xf32>) -> tensor<84x7x77x9xf32>
    %4 = tosa.reduce_min %1 {axis = 2 : i32} : (tensor<1x9x75xi1>) -> tensor<1x9x1xi1>
    %5 = tosa.ceil %3 : (tensor<84x7x77x9xf32>) -> tensor<84x7x77x9xf32>
    %in_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %6 = tosa.negate %2, %in_zp_6, %out_zp_6 : (tensor<1x9x1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1x9x1xi1>
    %7 = tosa.argmax %4 {axis = 1 : i32} : (tensor<1x9x1xi1>) -> tensor<1x1xi32>
    %8 = tosa.maximum %7, %7 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
    %9 = tosa.rsqrt %3 : (tensor<84x7x77x9xf32>) -> tensor<84x7x77x9xf32>
    %10 = tosa.bitwise_not %8 : (tensor<1x1xi32>) -> tensor<1x1xi32>
    %11 = tosa.reduce_any %4 {axis = 1 : i32} : (tensor<1x9x1xi1>) -> tensor<1x1x1xi1>
    %12 = tosa.intdiv %7, %7 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
    %in_zp_13 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_13 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %13 = tosa.negate %12, %in_zp_13, %out_zp_13 : (tensor<1x1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1xi32>
    %14 = tosa.logical_and %4, %4 : (tensor<1x9x1xi1>, tensor<1x9x1xi1>) -> tensor<1x9x1xi1>
    %15 = tosa.floor %9 : (tensor<84x7x77x9xf32>) -> tensor<84x7x77x9xf32>
    %16 = tosa.bitwise_xor %14, %6 : (tensor<1x9x1xi1>, tensor<1x9x1xi1>) -> tensor<1x9x1xi1>
    return %0, %5, %10, %11, %13, %15, %16 : tensor<77x99x56xi32>, tensor<84x7x77x9xf32>, tensor<1x1xi32>, tensor<1x1x1xi1>, tensor<1x1xi32>, tensor<84x7x77x9xf32>, tensor<1x9x1xi1>
  }
}
