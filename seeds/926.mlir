module {
  func.func @main(%arg0: tensor<74x42x61xi1>, %arg1: tensor<19x28x18x74xi32>, %arg2: tensor<1x1x1x1xi32>, %arg3: tensor<39x64x57x10xf32>, %arg4: tensor<39x1x57x1xf32>) -> (tensor<222x1x3xi1>, tensor<19x28x18x1xi32>, tensor<19x28x18x74xi1>, tensor<19x1x18x74xi1>, tensor<19x28x18x74xi1>, tensor<39x64x57x10xf32>) {
    %0 = tosa.reduce_all %arg0 {axis = 2 : i32} : (tensor<74x42x61xi1>) -> tensor<74x42x1xi1>
    %1 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<74x42x1xi1>) -> tensor<74x1x1xi1>
    %t_2 = tosa.const_shape {values = dense<[ 3, 1, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.tile %1, %t_2 : (tensor<74x1x1xi1>, !tosa.shape<3>) -> tensor<222x1x3xi1>
    %3 = tosa.reduce_max %2 {axis = 1 : i32} : (tensor<222x1x3xi1>) -> tensor<222x1x3xi1>
    %4 = tosa.abs %3 : (tensor<222x1x3xi1>) -> tensor<222x1x3xi1>
    %5 = tosa.intdiv %arg1, %arg2 : (tensor<19x28x18x74xi32>, tensor<1x1x1x1xi32>) -> tensor<19x28x18x74xi32>
    %6 = tosa.intdiv %5, %5 : (tensor<19x28x18x74xi32>, tensor<19x28x18x74xi32>) -> tensor<19x28x18x74xi32>
    %7 = tosa.pow %arg3, %arg4 : (tensor<39x64x57x10xf32>, tensor<39x1x57x1xf32>) -> tensor<39x64x57x10xf32>
    %8 = tosa.equal %5, %6 : (tensor<19x28x18x74xi32>, tensor<19x28x18x74xi32>) -> tensor<19x28x18x74xi1>
    %9 = tosa.reduce_min %5 {axis = 3 : i32} : (tensor<19x28x18x74xi32>) -> tensor<19x28x18x1xi32>
    %10 = tosa.add %5, %5 : (tensor<19x28x18x74xi32>, tensor<19x28x18x74xi32>) -> tensor<19x28x18x74xi32>
    %in_zp_11 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_11 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %11 = tosa.negate %5, %in_zp_11, %out_zp_11 : (tensor<19x28x18x74xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<19x28x18x74xi32>
    %12 = tosa.equal %11, %10 : (tensor<19x28x18x74xi32>, tensor<19x28x18x74xi32>) -> tensor<19x28x18x74xi1>
    %13 = tosa.logical_xor %12, %8 : (tensor<19x28x18x74xi1>, tensor<19x28x18x74xi1>) -> tensor<19x28x18x74xi1>
    %14 = tosa.bitwise_or %12, %12 : (tensor<19x28x18x74xi1>, tensor<19x28x18x74xi1>) -> tensor<19x28x18x74xi1>
    %15 = tosa.reduce_any %12 {axis = 1 : i32} : (tensor<19x28x18x74xi1>) -> tensor<19x1x18x74xi1>
    %16 = tosa.add %12, %13 : (tensor<19x28x18x74xi1>, tensor<19x28x18x74xi1>) -> tensor<19x28x18x74xi1>
    %17 = tosa.sigmoid %7 : (tensor<39x64x57x10xf32>) -> tensor<39x64x57x10xf32>
    return %4, %9, %14, %15, %16, %17 : tensor<222x1x3xi1>, tensor<19x28x18x1xi32>, tensor<19x28x18x74xi1>, tensor<19x1x18x74xi1>, tensor<19x28x18x74xi1>, tensor<39x64x57x10xf32>
  }
}
