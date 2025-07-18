module {
  func.func @main(%arg0: tensor<90x96x21x75x53x4xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<75x30x33xi8>, %arg4: tensor<37x12x26xi1>) -> (tensor<i32>, tensor<1x30x33xi8>, tensor<90x96x21x75x53x4xf32>, tensor<90x96x21x75x53x4xf32>, tensor<2x30x33xi8>, tensor<90x96x21x75x53x4xf32>, tensor<90x96x21x75x53x4xf32>, tensor<37x12x26xi1>, tensor<30x33xi1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<90x96x21x75x53x4xf32>) -> tensor<90x96x21x75x53x4xf32>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tosa.reduce_sum %arg3 {axis = 0 : i32} : (tensor<75x30x33xi8>) -> tensor<1x30x33xi8>
    %3 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<1x30x33xi8>) -> tensor<1x30x33xi8>
    %4 = tosa.clamp %3 {min_val = -29 : i8, max_val = 54 : i8} : (tensor<1x30x33xi8>) -> tensor<1x30x33xi8>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<1x30x33xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x30x33xi8>
    %6 = tosa.bitwise_or %5, %4 : (tensor<1x30x33xi8>, tensor<1x30x33xi8>) -> tensor<1x30x33xi8>
    %7 = tosa.bitwise_or %2, %5 : (tensor<1x30x33xi8>, tensor<1x30x33xi8>) -> tensor<1x30x33xi8>
    %8 = tosa.floor %0 : (tensor<90x96x21x75x53x4xf32>) -> tensor<90x96x21x75x53x4xf32>
    %9 = tosa.rsqrt %0 : (tensor<90x96x21x75x53x4xf32>) -> tensor<90x96x21x75x53x4xf32>
    %10 = tosa.reverse %6 {axis = 1 : i32} : (tensor<1x30x33xi8>) -> tensor<1x30x33xi8>
    %11 = tosa.reciprocal %0 : (tensor<90x96x21x75x53x4xf32>) -> tensor<90x96x21x75x53x4xf32>
    %12 = tosa.concat %2, %6 {axis = 0 : i32} : (tensor<1x30x33xi8>, tensor<1x30x33xi8>) -> tensor<2x30x33xi8>
    %13 = tosa.sigmoid %8 : (tensor<90x96x21x75x53x4xf32>) -> tensor<90x96x21x75x53x4xf32>
    %in_zp_14 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_14 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %14 = tosa.negate %12, %in_zp_14, %out_zp_14 : (tensor<2x30x33xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2x30x33xi8>
    %15 = tosa.argmax %10 {axis = 0 : i32} : (tensor<1x30x33xi8>) -> tensor<30x33xi32>
    %16 = tosa.logical_not %arg4 : (tensor<37x12x26xi1>) -> tensor<37x12x26xi1>
    %17 = tosa.log %11 : (tensor<90x96x21x75x53x4xf32>) -> tensor<90x96x21x75x53x4xf32>
    %18 = tosa.minimum %15, %15 : (tensor<30x33xi32>, tensor<30x33xi32>) -> tensor<30x33xi32>
    %19 = tosa.exp %8 : (tensor<90x96x21x75x53x4xf32>) -> tensor<90x96x21x75x53x4xf32>
    %20 = tosa.sub %16, %16 : (tensor<37x12x26xi1>, tensor<37x12x26xi1>) -> tensor<37x12x26xi1>
    %21 = tosa.greater %18, %18 : (tensor<30x33xi32>, tensor<30x33xi32>) -> tensor<30x33xi1>
    return %1, %7, %9, %13, %14, %17, %19, %20, %21 : tensor<i32>, tensor<1x30x33xi8>, tensor<90x96x21x75x53x4xf32>, tensor<90x96x21x75x53x4xf32>, tensor<2x30x33xi8>, tensor<90x96x21x75x53x4xf32>, tensor<90x96x21x75x53x4xf32>, tensor<37x12x26xi1>, tensor<30x33xi1>
  }
}
