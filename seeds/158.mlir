module {
  func.func @main(%arg0: tensor<9x97x23x45x1xi8>, %arg1: tensor<1x1x23x45x1xi8>, %arg2: tensor<93x75x35x26x76xf32>, %arg3: tensor<8x11x97x42xi1>) -> (tensor<9x97x23x45x1xi8>, tensor<93x75x35x26x76xf32>, tensor<8x11x1x1xi1>, tensor<8x11x1x42xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<9x97x23x45x1xi8>, tensor<1x1x23x45x1xi8>) -> tensor<9x97x23x45x1xi8>
    %1 = tosa.ceil %arg2 : (tensor<93x75x35x26x76xf32>) -> tensor<93x75x35x26x76xf32>
    %2 = tosa.sigmoid %1 : (tensor<93x75x35x26x76xf32>) -> tensor<93x75x35x26x76xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<93x75x35x26x76xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<93x75x35x26x76xf32>
    %4 = tosa.floor %3 : (tensor<93x75x35x26x76xf32>) -> tensor<93x75x35x26x76xf32>
    %5 = tosa.reduce_product %arg3 {axis = 2 : i32} : (tensor<8x11x97x42xi1>) -> tensor<8x11x1x42xi1>
    %6 = tosa.logical_xor %5, %5 : (tensor<8x11x1x42xi1>, tensor<8x11x1x42xi1>) -> tensor<8x11x1x42xi1>
    %7 = tosa.reduce_max %5 {axis = 3 : i32} : (tensor<8x11x1x42xi1>) -> tensor<8x11x1x1xi1>
    %8 = tosa.bitwise_xor %6, %5 : (tensor<8x11x1x42xi1>, tensor<8x11x1x42xi1>) -> tensor<8x11x1x42xi1>
    return %0, %4, %7, %8 : tensor<9x97x23x45x1xi8>, tensor<93x75x35x26x76xf32>, tensor<8x11x1x1xi1>, tensor<8x11x1x42xi1>
  }
}
