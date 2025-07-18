module {
  func.func @main(%arg0: tensor<99x21xi1>, %arg1: tensor<43x72x7x66xi64>, %arg2: tensor<1x72x1x1xi64>, %arg3: tensor<68x98x74xi64>, %arg4: tensor<68x98x1xi64>, %arg5: tensor<18xf32>) -> (tensor<43x144x7x66xi1>, tensor<4x3x7x11xi1>, tensor<68x98x74xi64>, tensor<18xf32>, tensor<1xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<99x21xi1>) -> tensor<99x21xi1>
    %1 = tosa.equal %arg1, %arg2 : (tensor<43x72x7x66xi64>, tensor<1x72x1x1xi64>) -> tensor<43x72x7x66xi1>
    %2 = tosa.bitwise_and %0, %0 : (tensor<99x21xi1>, tensor<99x21xi1>) -> tensor<99x21xi1>
    %3 = tosa.logical_or %1, %1 : (tensor<43x72x7x66xi1>, tensor<43x72x7x66xi1>) -> tensor<43x72x7x66xi1>
    %4 = tosa.bitwise_not %3 : (tensor<43x72x7x66xi1>) -> tensor<43x72x7x66xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %2, %in_zp_5, %out_zp_5 : (tensor<99x21xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<99x21xi1>
    %6 = tosa.logical_or %5, %2 : (tensor<99x21xi1>, tensor<99x21xi1>) -> tensor<99x21xi1>
    %r_7 = tosa.const_shape {values = dense<[ 2079 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %7 = tosa.reshape %6, %r_7 : (tensor<99x21xi1>, !tosa.shape<1>) -> tensor<2079xi1>
    %8 = tosa.bitwise_and %4, %3 : (tensor<43x72x7x66xi1>, tensor<43x72x7x66xi1>) -> tensor<43x72x7x66xi1>
    %9 = tosa.add %7, %7 : (tensor<2079xi1>, tensor<2079xi1>) -> tensor<2079xi1>
    %10 = tosa.bitwise_xor %9, %9 : (tensor<2079xi1>, tensor<2079xi1>) -> tensor<2079xi1>
    %11 = tosa.abs %8 : (tensor<43x72x7x66xi1>) -> tensor<43x72x7x66xi1>
    %12 = tosa.logical_and %10, %7 : (tensor<2079xi1>, tensor<2079xi1>) -> tensor<2079xi1>
    %13 = tosa.reverse %12 {axis = 0 : i32} : (tensor<2079xi1>) -> tensor<2079xi1>
    %14 = tosa.bitwise_and %11, %11 : (tensor<43x72x7x66xi1>, tensor<43x72x7x66xi1>) -> tensor<43x72x7x66xi1>
    %r_15 = tosa.const_shape {values = dense<[ 1, 33, 3, 21 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %15 = tosa.reshape %13, %r_15 : (tensor<2079xi1>, !tosa.shape<4>) -> tensor<1x33x3x21xi1>
    %16 = tosa.concat %14, %8 {axis = 1 : i32} : (tensor<43x72x7x66xi1>, tensor<43x72x7x66xi1>) -> tensor<43x144x7x66xi1>
    %17 = tosa.maximum %arg3, %arg4 : (tensor<68x98x74xi64>, tensor<68x98x1xi64>) -> tensor<68x98x74xi64>
    %18 = tosa.reduce_max %15 {axis = 2 : i32} : (tensor<1x33x3x21xi1>) -> tensor<1x33x1x21xi1>
    %19 = tosa.add %17, %17 : (tensor<68x98x74xi64>, tensor<68x98x74xi64>) -> tensor<68x98x74xi64>
    %s_20_start = tosa.const_shape {values = dense<[ 0, 1, 0, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_20_size = tosa.const_shape {values = dense<[ 4, 3, 7, 11 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %20 = tosa.slice %18, %s_20_start, %s_20_size : (tensor<1x33x1x21xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<4x3x7x11xi1>
    %21 = tosa.rsqrt %arg5 : (tensor<18xf32>) -> tensor<18xf32>
    %in_zp_22 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_22 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %22 = tosa.negate %21, %in_zp_22, %out_zp_22 : (tensor<18xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<18xf32>
    %23 = tosa.bitwise_xor %19, %17 : (tensor<68x98x74xi64>, tensor<68x98x74xi64>) -> tensor<68x98x74xi64>
    %24 = tosa.add %20, %20 : (tensor<4x3x7x11xi1>, tensor<4x3x7x11xi1>) -> tensor<4x3x7x11xi1>
    %25 = tosa.bitwise_or %23, %19 : (tensor<68x98x74xi64>, tensor<68x98x74xi64>) -> tensor<68x98x74xi64>
    %26 = tosa.reciprocal %22 : (tensor<18xf32>) -> tensor<18xf32>
    %27 = tosa.maximum %26, %26 : (tensor<18xf32>, tensor<18xf32>) -> tensor<18xf32>
    %28 = tosa.reduce_max %22 {axis = 0 : i32} : (tensor<18xf32>) -> tensor<1xf32>
    return %16, %24, %25, %27, %28 : tensor<43x144x7x66xi1>, tensor<4x3x7x11xi1>, tensor<68x98x74xi64>, tensor<18xf32>, tensor<1xf32>
  }
}
