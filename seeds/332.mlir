module {
  func.func @main(%arg0: tensor<65x61x20xi64>, %arg1: tensor<65x20x40xi64>, %arg2: tensor<26x6xi1>, %arg3: tensor<26x6xi1>, %arg4: tensor<70x33x52x10xf32>, %arg5: tensor<i32>, %arg6: tensor<i32>) -> (tensor<1x6xi1>, tensor<70x33x52x10xf32>, tensor<65x61x40xi1>, tensor<70x33x52x10xf32>, tensor<65x1x40xi1>, tensor<26x1xi1>, tensor<26x6xi1>, tensor<70x1x52x10xf32>, tensor<70x33x52x10xf32>, tensor<i32>, tensor<1x6xi1>, tensor<70x1x52x10xf32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<65x61x20xi64>, tensor<65x20x40xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<65x61x40xi64>
    %1 = tosa.maximum %0, %0 : (tensor<65x61x40xi64>, tensor<65x61x40xi64>) -> tensor<65x61x40xi64>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<26x6xi1>, tensor<26x6xi1>) -> tensor<26x6xi1>
    %3 = tosa.logical_and %2, %2 : (tensor<26x6xi1>, tensor<26x6xi1>) -> tensor<26x6xi1>
    %4 = tosa.logical_xor %3, %3 : (tensor<26x6xi1>, tensor<26x6xi1>) -> tensor<26x6xi1>
    %5 = tosa.equal %0, %1 : (tensor<65x61x40xi64>, tensor<65x61x40xi64>) -> tensor<65x61x40xi1>
    %6 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<26x6xi1>) -> tensor<1x6xi1>
    %7 = tosa.reduce_product %4 {axis = 1 : i32} : (tensor<26x6xi1>) -> tensor<26x1xi1>
    %8 = tosa.logical_and %6, %6 : (tensor<1x6xi1>, tensor<1x6xi1>) -> tensor<1x6xi1>
    %9 = tosa.logical_not %8 : (tensor<1x6xi1>) -> tensor<1x6xi1>
    %10 = tosa.exp %arg4 : (tensor<70x33x52x10xf32>) -> tensor<70x33x52x10xf32>
    %11 = tosa.sub %10, %10 : (tensor<70x33x52x10xf32>, tensor<70x33x52x10xf32>) -> tensor<70x33x52x10xf32>
    %12 = tosa.maximum %10, %10 : (tensor<70x33x52x10xf32>, tensor<70x33x52x10xf32>) -> tensor<70x33x52x10xf32>
    %13 = tosa.logical_not %5 : (tensor<65x61x40xi1>) -> tensor<65x61x40xi1>
    %14 = tosa.sigmoid %12 : (tensor<70x33x52x10xf32>) -> tensor<70x33x52x10xf32>
    %15 = tosa.reverse %10 {axis = 2 : i32} : (tensor<70x33x52x10xf32>) -> tensor<70x33x52x10xf32>
    %16 = tosa.reduce_any %5 {axis = 1 : i32} : (tensor<65x61x40xi1>) -> tensor<65x1x40xi1>
    %17 = tosa.reduce_sum %10 {axis = 1 : i32} : (tensor<70x33x52x10xf32>) -> tensor<70x1x52x10xf32>
    %18 = tosa.clz %7 : (tensor<26x1xi1>) -> tensor<26x1xi1>
    %19 = tosa.clz %2 : (tensor<26x6xi1>) -> tensor<26x6xi1>
    %20 = tosa.minimum %14, %10 : (tensor<70x33x52x10xf32>, tensor<70x33x52x10xf32>) -> tensor<70x33x52x10xf32>
    %21 = tosa.rsqrt %17 : (tensor<70x1x52x10xf32>) -> tensor<70x1x52x10xf32>
    %22 = tosa.intdiv %arg5, %arg6 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %23 = tosa.floor %20 : (tensor<70x33x52x10xf32>) -> tensor<70x33x52x10xf32>
    %in_zp_24 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_24 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %24 = tosa.negate %17, %in_zp_24, %out_zp_24 : (tensor<70x1x52x10xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<70x1x52x10xf32>
    %25 = tosa.bitwise_or %22, %22 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %26 = tosa.logical_xor %6, %6 : (tensor<1x6xi1>, tensor<1x6xi1>) -> tensor<1x6xi1>
    %27 = tosa.clamp %24 {min_val = -2.200000e+01 : f32, max_val = 1.600000e+01 : f32} : (tensor<70x1x52x10xf32>) -> tensor<70x1x52x10xf32>
    return %9, %11, %13, %15, %16, %18, %19, %21, %23, %25, %26, %27 : tensor<1x6xi1>, tensor<70x33x52x10xf32>, tensor<65x61x40xi1>, tensor<70x33x52x10xf32>, tensor<65x1x40xi1>, tensor<26x1xi1>, tensor<26x6xi1>, tensor<70x1x52x10xf32>, tensor<70x33x52x10xf32>, tensor<i32>, tensor<1x6xi1>, tensor<70x1x52x10xf32>
  }
}
