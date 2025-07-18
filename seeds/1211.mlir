module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<68x70x58xf32>, %arg3: tensor<39x6x10x45xi1>) -> (tensor<i32>, tensor<1x70x58xf32>, tensor<552160x1xf32>, tensor<1x1x1x45xi1>, tensor<1x6x45xi1>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = tosa.tanh %arg2 : (tensor<68x70x58xf32>) -> tensor<68x70x58xf32>
    %2 = tosa.tanh %1 : (tensor<68x70x58xf32>) -> tensor<68x70x58xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<68x70x58xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<68x70x58xf32>
    %r_4 = tosa.const_shape {values = dense<[ 276080, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.reshape %3, %r_4 : (tensor<68x70x58xf32>, !tosa.shape<2>) -> tensor<276080x1xf32>
    %5 = tosa.concat %4, %4 {axis = 0 : i32} : (tensor<276080x1xf32>, tensor<276080x1xf32>) -> tensor<552160x1xf32>
    %6 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<68x70x58xf32>) -> tensor<1x70x58xf32>
    %7 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<39x6x10x45xi1>) -> tensor<1x6x10x45xi1>
    %8 = tosa.argmax %7 {axis = 2 : i32} : (tensor<1x6x10x45xi1>) -> tensor<1x6x45xi32>
    %9 = tosa.logical_left_shift %8, %8 : (tensor<1x6x45xi32>, tensor<1x6x45xi32>) -> tensor<1x6x45xi32>
    %10 = tosa.sigmoid %5 : (tensor<552160x1xf32>) -> tensor<552160x1xf32>
    %11 = tosa.bitwise_not %7 : (tensor<1x6x10x45xi1>) -> tensor<1x6x10x45xi1>
    %12 = tosa.reduce_all %11 {axis = 1 : i32} : (tensor<1x6x10x45xi1>) -> tensor<1x1x10x45xi1>
    %13 = tosa.reduce_all %12 {axis = 2 : i32} : (tensor<1x1x10x45xi1>) -> tensor<1x1x1x45xi1>
    %14 = tosa.equal %9, %9 : (tensor<1x6x45xi32>, tensor<1x6x45xi32>) -> tensor<1x6x45xi1>
    return %0, %6, %10, %13, %14 : tensor<i32>, tensor<1x70x58xf32>, tensor<552160x1xf32>, tensor<1x1x1x45xi1>, tensor<1x6x45xi1>
  }
}
