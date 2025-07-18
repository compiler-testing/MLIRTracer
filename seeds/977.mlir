module {
  func.func @main(%arg0: tensor<77xi64>, %arg1: tensor<23xi1>, %arg2: tensor<23xi1>, %arg3: tensor<94x57xf32>, %arg4: tensor<21xi32>, %arg5: tensor<1xi32>) -> (tensor<3xi64>, tensor<94x57xf32>, tensor<1xi1>, tensor<94x57xf32>, tensor<94x57xf32>, tensor<94x57xf32>, tensor<1xi64>, tensor<94x1xf32>, tensor<94x57xf32>, tensor<94x57xf32>, tensor<94x57xf32>, tensor<21xi32>) {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<77xi64>) -> tensor<1xi64>
    %1 = tosa.bitwise_and %0, %0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %2 = tosa.logical_xor %arg1, %arg2 : (tensor<23xi1>, tensor<23xi1>) -> tensor<23xi1>
    %3 = tosa.ceil %arg3 : (tensor<94x57xf32>) -> tensor<94x57xf32>
    %4 = tosa.arithmetic_right_shift %2, %2 {round = true} : (tensor<23xi1>, tensor<23xi1>) -> tensor<23xi1>
    %5 = tosa.bitwise_or %1, %1 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %t_6 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.tile %5, %t_6 : (tensor<1xi64>, !tosa.shape<1>) -> tensor<3xi64>
    %7 = tosa.reciprocal %3 : (tensor<94x57xf32>) -> tensor<94x57xf32>
    %8 = tosa.reduce_all %4 {axis = 0 : i32} : (tensor<23xi1>) -> tensor<1xi1>
    %9 = tosa.logical_not %8 : (tensor<1xi1>) -> tensor<1xi1>
    %10 = tosa.tanh %3 : (tensor<94x57xf32>) -> tensor<94x57xf32>
    %11 = tosa.minimum %10, %3 : (tensor<94x57xf32>, tensor<94x57xf32>) -> tensor<94x57xf32>
    %12 = tosa.pow %3, %3 : (tensor<94x57xf32>, tensor<94x57xf32>) -> tensor<94x57xf32>
    %13 = tosa.abs %11 : (tensor<94x57xf32>) -> tensor<94x57xf32>
    %14 = tosa.sub %12, %3 : (tensor<94x57xf32>, tensor<94x57xf32>) -> tensor<94x57xf32>
    %15 = tosa.log %12 : (tensor<94x57xf32>) -> tensor<94x57xf32>
    %16 = tosa.minimum %12, %15 : (tensor<94x57xf32>, tensor<94x57xf32>) -> tensor<94x57xf32>
    %17 = tosa.ceil %12 : (tensor<94x57xf32>) -> tensor<94x57xf32>
    %18 = tosa.bitwise_or %1, %1 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %19 = tosa.reduce_max %12 {axis = 1 : i32} : (tensor<94x57xf32>) -> tensor<94x1xf32>
    %20 = tosa.intdiv %arg4, %arg5 : (tensor<21xi32>, tensor<1xi32>) -> tensor<21xi32>
    %21 = tosa.reverse %13 {axis = 1 : i32} : (tensor<94x57xf32>) -> tensor<94x57xf32>
    %22 = tosa.floor %12 : (tensor<94x57xf32>) -> tensor<94x57xf32>
    %23 = tosa.clamp %12 {min_val = 1.600000e+01 : f32, max_val = 1.300000e+02 : f32} : (tensor<94x57xf32>) -> tensor<94x57xf32>
    %24 = tosa.arithmetic_right_shift %20, %20 {round = true} : (tensor<21xi32>, tensor<21xi32>) -> tensor<21xi32>
    return %6, %7, %9, %14, %16, %17, %18, %19, %21, %22, %23, %24 : tensor<3xi64>, tensor<94x57xf32>, tensor<1xi1>, tensor<94x57xf32>, tensor<94x57xf32>, tensor<94x57xf32>, tensor<1xi64>, tensor<94x1xf32>, tensor<94x57xf32>, tensor<94x57xf32>, tensor<94x57xf32>, tensor<21xi32>
  }
}
