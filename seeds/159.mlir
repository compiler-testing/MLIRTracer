module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<44xi8>, %arg3: tensor<29x29x1xf32>) -> (tensor<i8>, tensor<29x29x1xf32>, tensor<3xi1>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %1 = tosa.reduce_max %arg2 {axis = 0 : i32} : (tensor<44xi8>) -> tensor<1xi8>
    %2 = tosa.bitwise_and %0, %0 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %3 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %4 = tosa.transpose %1 {perms = array<i32: 0>} : (tensor<1xi8>) -> tensor<1xi8>
    %5 = tosa.sigmoid %arg3 : (tensor<29x29x1xf32>) -> tensor<29x29x1xf32>
    %6 = tosa.maximum %5, %5 : (tensor<29x29x1xf32>, tensor<29x29x1xf32>) -> tensor<29x29x1xf32>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = tosa.negate %4, %in_zp_7, %out_zp_7 : (tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %8 = tosa.add %5, %6 : (tensor<29x29x1xf32>, tensor<29x29x1xf32>) -> tensor<29x29x1xf32>
    %9 = tosa.reciprocal %8 : (tensor<29x29x1xf32>) -> tensor<29x29x1xf32>
    %10 = tosa.greater %7, %1 : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi1>
    %t_11 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %11 = tosa.tile %10, %t_11 : (tensor<1xi1>, !tosa.shape<1>) -> tensor<3xi1>
    %12 = tosa.bitwise_xor %11, %11 : (tensor<3xi1>, tensor<3xi1>) -> tensor<3xi1>
    return %2, %9, %12 : tensor<i8>, tensor<29x29x1xf32>, tensor<3xi1>
  }
}
