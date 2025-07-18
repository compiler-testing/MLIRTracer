module {
  func.func @main(%arg0: tensor<69x8xi64>, %arg1: tensor<6xi1>, %arg2: tensor<6xi1>, %arg3: tensor<55x37x10xf32>) -> (tensor<1x1xi64>, tensor<1xi1>, tensor<55x37x10xf32>) {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<69x8xi64>) -> tensor<1x8xi64>
    %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x8xi64>) -> tensor<1x1xi64>
    %r_2 = tosa.const_shape {values = dense<[ 1, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.reshape %1, %r_2 : (tensor<1x1xi64>, !tosa.shape<4>) -> tensor<1x1x1x1xi64>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<1x1x1x1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x1x1xi64>
    %r_4 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.reshape %3, %r_4 : (tensor<1x1x1x1xi64>, !tosa.shape<2>) -> tensor<1x1xi64>
    %5 = tosa.bitwise_or %4, %1 : (tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x1xi64>
    %6 = tosa.logical_and %arg1, %arg2 : (tensor<6xi1>, tensor<6xi1>) -> tensor<6xi1>
    %7 = tosa.reduce_any %6 {axis = 0 : i32} : (tensor<6xi1>) -> tensor<1xi1>
    %8 = tosa.ceil %arg3 : (tensor<55x37x10xf32>) -> tensor<55x37x10xf32>
    return %5, %7, %8 : tensor<1x1xi64>, tensor<1xi1>, tensor<55x37x10xf32>
  }
}
