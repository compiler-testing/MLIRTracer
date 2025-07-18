module {
  func.func @main(%arg0: tensor<45x93x12x66x6x58xi1>, %arg1: tensor<45x93x1x1x1x1xi1>, %arg2: tensor<13x36xi8>, %arg3: tensor<8x26x28x2x40xf32>) -> (tensor<45x93x12x66x6x58xi1>, tensor<13x1xi8>, tensor<8x26x28x2x40xf32>, tensor<45x93x12x66x6x58xi1>, tensor<13x36xi8>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<45x93x12x66x6x58xi1>, tensor<45x93x1x1x1x1xi1>) -> tensor<45x93x12x66x6x58xi1>
    %1 = tosa.reverse %arg2 {axis = 1 : i32} : (tensor<13x36xi8>) -> tensor<13x36xi8>
    %2 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<13x36xi8>) -> tensor<13x1xi8>
    %3 = tosa.identity %2 : (tensor<13x1xi8>) -> tensor<13x1xi8>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<13x1xi8>, tensor<13x1xi8>) -> tensor<13x1xi8>
    %5 = tosa.clz %4 : (tensor<13x1xi8>) -> tensor<13x1xi8>
    %6 = tosa.logical_or %0, %0 : (tensor<45x93x12x66x6x58xi1>, tensor<45x93x12x66x6x58xi1>) -> tensor<45x93x12x66x6x58xi1>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = tosa.negate %1, %in_zp_7, %out_zp_7 : (tensor<13x36xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x36xi8>
    %8 = tosa.bitwise_or %5, %5 : (tensor<13x1xi8>, tensor<13x1xi8>) -> tensor<13x1xi8>
    %9 = tosa.reciprocal %arg3 : (tensor<8x26x28x2x40xf32>) -> tensor<8x26x28x2x40xf32>
    %10 = tosa.logical_not %0 : (tensor<45x93x12x66x6x58xi1>) -> tensor<45x93x12x66x6x58xi1>
    %11 = tosa.bitwise_or %7, %7 : (tensor<13x36xi8>, tensor<13x36xi8>) -> tensor<13x36xi8>
    return %6, %8, %9, %10, %11 : tensor<45x93x12x66x6x58xi1>, tensor<13x1xi8>, tensor<8x26x28x2x40xf32>, tensor<45x93x12x66x6x58xi1>, tensor<13x36xi8>
  }
}
