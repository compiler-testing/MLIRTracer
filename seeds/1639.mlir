module {
  func.func @main(%arg0: tensor<49x92x25x86x28x60xf32>, %arg1: tensor<96x61x92x31xi8>, %arg2: tensor<i1>, %arg3: tensor<i1>, %arg4: tensor<42x3x10x38xi32>, %arg5: tensor<42x1x1x38xi32>, %arg6: tensor<85x37xi1>) -> (tensor<96x61x92x31xi8>, tensor<744x46x488x1xi8>, tensor<85x1xi1>, tensor<i1>, tensor<49x92x25x86x28x60xf32>, tensor<42x3x10x38xi1>, tensor<42x3x10x38xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<49x92x25x86x28x60xf32>) -> tensor<49x92x25x86x28x60xf32>
    %1 = tosa.clz %arg1 : (tensor<96x61x92x31xi8>) -> tensor<96x61x92x31xi8>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<96x61x92x31xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<96x61x92x31xi8>
    %3 = tosa.tanh %0 : (tensor<49x92x25x86x28x60xf32>) -> tensor<49x92x25x86x28x60xf32>
    %4 = tosa.arithmetic_right_shift %2, %1 {round = true} : (tensor<96x61x92x31xi8>, tensor<96x61x92x31xi8>) -> tensor<96x61x92x31xi8>
    %5 = tosa.bitwise_not %2 : (tensor<96x61x92x31xi8>) -> tensor<96x61x92x31xi8>
    %6 = tosa.clamp %4 {min_val = 29 : i8, max_val = 51 : i8} : (tensor<96x61x92x31xi8>) -> tensor<96x61x92x31xi8>
    %7 = tosa.bitwise_and %5, %6 : (tensor<96x61x92x31xi8>, tensor<96x61x92x31xi8>) -> tensor<96x61x92x31xi8>
    %8 = tosa.logical_xor %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %9 = tosa.intdiv %arg4, %arg5 : (tensor<42x3x10x38xi32>, tensor<42x1x1x38xi32>) -> tensor<42x3x10x38xi32>
    %r_10 = tosa.const_shape {values = dense<[ 744, 46, 488, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.reshape %6, %r_10 : (tensor<96x61x92x31xi8>, !tosa.shape<4>) -> tensor<744x46x488x1xi8>
    %11 = tosa.reduce_any %arg6 {axis = 1 : i32} : (tensor<85x37xi1>) -> tensor<85x1xi1>
    %12 = tosa.bitwise_and %9, %9 : (tensor<42x3x10x38xi32>, tensor<42x3x10x38xi32>) -> tensor<42x3x10x38xi32>
    %13 = tosa.logical_or %8, %8 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %14 = tosa.sigmoid %3 : (tensor<49x92x25x86x28x60xf32>) -> tensor<49x92x25x86x28x60xf32>
    %15 = tosa.ceil %14 : (tensor<49x92x25x86x28x60xf32>) -> tensor<49x92x25x86x28x60xf32>
    %16 = tosa.greater_equal %9, %12 : (tensor<42x3x10x38xi32>, tensor<42x3x10x38xi32>) -> tensor<42x3x10x38xi1>
    %17 = tosa.reverse %16 {axis = 2 : i32} : (tensor<42x3x10x38xi1>) -> tensor<42x3x10x38xi1>
    %18 = tosa.bitwise_and %17, %17 : (tensor<42x3x10x38xi1>, tensor<42x3x10x38xi1>) -> tensor<42x3x10x38xi1>
    %in_zp_19 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_19 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %19 = tosa.negate %16, %in_zp_19, %out_zp_19 : (tensor<42x3x10x38xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<42x3x10x38xi1>
    return %7, %10, %11, %13, %15, %18, %19 : tensor<96x61x92x31xi8>, tensor<744x46x488x1xi8>, tensor<85x1xi1>, tensor<i1>, tensor<49x92x25x86x28x60xf32>, tensor<42x3x10x38xi1>, tensor<42x3x10x38xi1>
  }
}
