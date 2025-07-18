module {
  func.func @main(%arg0: tensor<47xi32>, %arg1: tensor<64xf32>, %arg2: tensor<59x73x87x47x25x28xi1>, %arg3: tensor<1x1x1x1x1x1xi1>) -> (tensor<1xi32>, tensor<3xf32>, tensor<59x73x87x47x25x28xi1>, tensor<128xf32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<47xi32>) -> tensor<1xi32>
    %1 = tosa.sigmoid %arg1 : (tensor<64xf32>) -> tensor<64xf32>
    %2 = tosa.bitwise_or %0, %0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %3 = tosa.clz %2 : (tensor<1xi32>) -> tensor<1xi32>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = tosa.negate %3, %in_zp_4, %out_zp_4 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = tosa.logical_right_shift %4, %2 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %6 = tosa.reciprocal %1 : (tensor<64xf32>) -> tensor<64xf32>
    %s_7_start = tosa.const_shape {values = dense<[ 61 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_7_size = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %7 = tosa.slice %6, %s_7_start, %s_7_size : (tensor<64xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf32>
    %t_8 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %8 = tosa.tile %1, %t_8 : (tensor<64xf32>, !tosa.shape<1>) -> tensor<128xf32>
    %9 = tosa.logical_or %arg2, %arg3 : (tensor<59x73x87x47x25x28xi1>, tensor<1x1x1x1x1x1xi1>) -> tensor<59x73x87x47x25x28xi1>
    %10 = tosa.maximum %7, %7 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %11 = tosa.logical_not %9 : (tensor<59x73x87x47x25x28xi1>) -> tensor<59x73x87x47x25x28xi1>
    %12 = tosa.logical_and %11, %11 : (tensor<59x73x87x47x25x28xi1>, tensor<59x73x87x47x25x28xi1>) -> tensor<59x73x87x47x25x28xi1>
    %13 = tosa.floor %8 : (tensor<128xf32>) -> tensor<128xf32>
    return %5, %10, %12, %13 : tensor<1xi32>, tensor<3xf32>, tensor<59x73x87x47x25x28xi1>, tensor<128xf32>
  }
}
