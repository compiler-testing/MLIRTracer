module {
  func.func @main(%arg0: tensor<11x61x77x59x65xf32>, %arg1: tensor<97x80x89x39x63x35xi1>, %arg2: tensor<39xi64>, %arg3: tensor<63x100xi1>) -> (tensor<97x80x89x39x63x35xi1>, tensor<11x61x77x59x65xf32>, tensor<1xi64>, tensor<1x100xi1>, tensor<1x100xi1>, tensor<1xi64>, tensor<11x61x77x59x65xf32>, tensor<11x61x77x59x65xf32>, tensor<11x61x77x59x65xf32>) {
    %0 = tosa.floor %arg0 : (tensor<11x61x77x59x65xf32>) -> tensor<11x61x77x59x65xf32>
    %1 = tosa.reciprocal %0 : (tensor<11x61x77x59x65xf32>) -> tensor<11x61x77x59x65xf32>
    %2 = tosa.logical_not %arg1 : (tensor<97x80x89x39x63x35xi1>) -> tensor<97x80x89x39x63x35xi1>
    %3 = tosa.reduce_sum %arg2 {axis = 0 : i32} : (tensor<39xi64>) -> tensor<1xi64>
    %4 = tosa.bitwise_or %2, %2 : (tensor<97x80x89x39x63x35xi1>, tensor<97x80x89x39x63x35xi1>) -> tensor<97x80x89x39x63x35xi1>
    %5 = tosa.ceil %1 : (tensor<11x61x77x59x65xf32>) -> tensor<11x61x77x59x65xf32>
    %6 = tosa.exp %5 : (tensor<11x61x77x59x65xf32>) -> tensor<11x61x77x59x65xf32>
    %7 = tosa.abs %6 : (tensor<11x61x77x59x65xf32>) -> tensor<11x61x77x59x65xf32>
    %8 = tosa.reverse %3 {axis = 0 : i32} : (tensor<1xi64>) -> tensor<1xi64>
    %9 = tosa.logical_left_shift %3, %3 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %10 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<63x100xi1>) -> tensor<1x100xi1>
    %11 = tosa.logical_and %10, %10 : (tensor<1x100xi1>, tensor<1x100xi1>) -> tensor<1x100xi1>
    %12 = tosa.logical_right_shift %10, %10 : (tensor<1x100xi1>, tensor<1x100xi1>) -> tensor<1x100xi1>
    %in_zp_13 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_13 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %13 = tosa.negate %11, %in_zp_13, %out_zp_13 : (tensor<1x100xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1x100xi1>
    %14 = tosa.clz %12 : (tensor<1x100xi1>) -> tensor<1x100xi1>
    %in_zp_15 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_15 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %15 = tosa.negate %9, %in_zp_15, %out_zp_15 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %16 = tosa.pow %5, %0 : (tensor<11x61x77x59x65xf32>, tensor<11x61x77x59x65xf32>) -> tensor<11x61x77x59x65xf32>
    %17 = tosa.pow %5, %0 : (tensor<11x61x77x59x65xf32>, tensor<11x61x77x59x65xf32>) -> tensor<11x61x77x59x65xf32>
    %18 = tosa.log %1 : (tensor<11x61x77x59x65xf32>) -> tensor<11x61x77x59x65xf32>
    return %4, %7, %8, %13, %14, %15, %16, %17, %18 : tensor<97x80x89x39x63x35xi1>, tensor<11x61x77x59x65xf32>, tensor<1xi64>, tensor<1x100xi1>, tensor<1x100xi1>, tensor<1xi64>, tensor<11x61x77x59x65xf32>, tensor<11x61x77x59x65xf32>, tensor<11x61x77x59x65xf32>
  }
}
