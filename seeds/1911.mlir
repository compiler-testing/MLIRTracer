module {
  func.func @main(%arg0: tensor<90x32x33x99x87x11xi8>, %arg1: tensor<63x43x64x61xi1>, %arg2: tensor<58xf32>) -> (tensor<90x32x33x99x87x11xi8>, tensor<58xf32>, tensor<1x1x64x61xi1>, tensor<126x1x64x61xi1>, tensor<63x1x64x61xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<90x32x33x99x87x11xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<90x32x33x99x87x11xi8>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<90x32x33x99x87x11xi8>, tensor<90x32x33x99x87x11xi8>) -> tensor<90x32x33x99x87x11xi8>
    %2 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<63x43x64x61xi1>) -> tensor<63x1x64x61xi1>
    %3 = tosa.maximum %1, %0 : (tensor<90x32x33x99x87x11xi8>, tensor<90x32x33x99x87x11xi8>) -> tensor<90x32x33x99x87x11xi8>
    %4 = tosa.rsqrt %arg2 : (tensor<58xf32>) -> tensor<58xf32>
    %5 = tosa.pow %4, %4 : (tensor<58xf32>, tensor<58xf32>) -> tensor<58xf32>
    %6 = tosa.bitwise_and %2, %2 : (tensor<63x1x64x61xi1>, tensor<63x1x64x61xi1>) -> tensor<63x1x64x61xi1>
    %7 = tosa.reduce_any %6 {axis = 0 : i32} : (tensor<63x1x64x61xi1>) -> tensor<1x1x64x61xi1>
    %8 = tosa.concat %2, %2 {axis = 0 : i32} : (tensor<63x1x64x61xi1>, tensor<63x1x64x61xi1>) -> tensor<126x1x64x61xi1>
    %9 = tosa.bitwise_or %2, %6 : (tensor<63x1x64x61xi1>, tensor<63x1x64x61xi1>) -> tensor<63x1x64x61xi1>
    %10 = tosa.sub %9, %2 : (tensor<63x1x64x61xi1>, tensor<63x1x64x61xi1>) -> tensor<63x1x64x61xi1>
    return %3, %5, %7, %8, %10 : tensor<90x32x33x99x87x11xi8>, tensor<58xf32>, tensor<1x1x64x61xi1>, tensor<126x1x64x61xi1>, tensor<63x1x64x61xi1>
  }
}
