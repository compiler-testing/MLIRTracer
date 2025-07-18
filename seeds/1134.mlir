module {
  func.func @main(%arg0: tensor<15x19x90x76x39x92xi32>, %arg1: tensor<54x96xi8>, %arg2: tensor<8x35x98xi1>) -> (tensor<15x19x90x76x39x92xi32>, tensor<1xi32>, tensor<1x96xi8>, tensor<8x1x98xi1>, tensor<8x35x98xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<15x19x90x76x39x92xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<15x19x90x76x39x92xi32>
    %1 = tosa.reduce_max %arg1 {axis = 0 : i32} : (tensor<54x96xi8>) -> tensor<1x96xi8>
    %2 = tosa.bitwise_or %1, %1 : (tensor<1x96xi8>, tensor<1x96xi8>) -> tensor<1x96xi8>
    %3 = tosa.abs %0 : (tensor<15x19x90x76x39x92xi32>) -> tensor<15x19x90x76x39x92xi32>
    %4 = tosa.intdiv %3, %0 : (tensor<15x19x90x76x39x92xi32>, tensor<15x19x90x76x39x92xi32>) -> tensor<15x19x90x76x39x92xi32>
    %5 = tosa.argmax %2 {axis = 1 : i32} : (tensor<1x96xi8>) -> tensor<1xi32>
    %6 = tosa.add %5, %5 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %7 = tosa.minimum %2, %2 : (tensor<1x96xi8>, tensor<1x96xi8>) -> tensor<1x96xi8>
    %8 = tosa.logical_not %arg2 : (tensor<8x35x98xi1>) -> tensor<8x35x98xi1>
    %9 = tosa.reduce_any %8 {axis = 1 : i32} : (tensor<8x35x98xi1>) -> tensor<8x1x98xi1>
    %10 = tosa.reverse %8 {axis = 0 : i32} : (tensor<8x35x98xi1>) -> tensor<8x35x98xi1>
    return %4, %6, %7, %9, %10 : tensor<15x19x90x76x39x92xi32>, tensor<1xi32>, tensor<1x96xi8>, tensor<8x1x98xi1>, tensor<8x35x98xi1>
  }
}
