module {
  func.func @main(%arg0: tensor<19x17xi8>, %arg1: tensor<56x97x67x61x32xi1>, %arg2: tensor<1x1x67x61x1xi1>, %arg3: tensor<15x14x34x87x12x21xf32>) -> (tensor<38x3xi8>, tensor<56x97x67x61x64xi1>, tensor<15x14x34x87x12x21xf32>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<19x17xi8>) -> tensor<19x17xi8>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<56x97x67x61x32xi1>, tensor<1x1x67x61x1xi1>) -> tensor<56x97x67x61x32xi1>
    %2 = tosa.logical_or %1, %1 : (tensor<56x97x67x61x32xi1>, tensor<56x97x67x61x32xi1>) -> tensor<56x97x67x61x32xi1>
    %3 = tosa.abs %2 : (tensor<56x97x67x61x32xi1>) -> tensor<56x97x67x61x32xi1>
    %4 = tosa.rsqrt %arg3 : (tensor<15x14x34x87x12x21xf32>) -> tensor<15x14x34x87x12x21xf32>
    %in_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<15x14x34x87x12x21xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<15x14x34x87x12x21xf32>
    %6 = tosa.sigmoid %5 : (tensor<15x14x34x87x12x21xf32>) -> tensor<15x14x34x87x12x21xf32>
    %7 = tosa.concat %3, %2 {axis = 4 : i32} : (tensor<56x97x67x61x32xi1>, tensor<56x97x67x61x32xi1>) -> tensor<56x97x67x61x64xi1>
    %8 = tosa.logical_xor %7, %7 : (tensor<56x97x67x61x64xi1>, tensor<56x97x67x61x64xi1>) -> tensor<56x97x67x61x64xi1>
    %9 = tosa.bitwise_not %8 : (tensor<56x97x67x61x64xi1>) -> tensor<56x97x67x61x64xi1>
    %10 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<19x17xi8>) -> tensor<19x1xi8>
    %t_11 = tosa.const_shape {values = dense<[ 2, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %11 = tosa.tile %10, %t_11 : (tensor<19x1xi8>, !tosa.shape<2>) -> tensor<38x3xi8>
    %12 = tosa.logical_and %9, %8 : (tensor<56x97x67x61x64xi1>, tensor<56x97x67x61x64xi1>) -> tensor<56x97x67x61x64xi1>
    %13 = tosa.floor %6 : (tensor<15x14x34x87x12x21xf32>) -> tensor<15x14x34x87x12x21xf32>
    return %11, %12, %13 : tensor<38x3xi8>, tensor<56x97x67x61x64xi1>, tensor<15x14x34x87x12x21xf32>
  }
}
