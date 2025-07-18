module {
  func.func @main(%arg0: tensor<3x42x76x45xi16>, %arg1: tensor<i8>, %arg2: tensor<i8>, %arg3: tensor<33x31x79xf32>) -> (tensor<3x1x76x45xi16>, tensor<1x1x76x45xi16>, tensor<i1>, tensor<i1>, tensor<66x31x79xf32>, tensor<33x31xi32>) {
    %0 = tosa.reduce_product %arg0 {axis = 1 : i32} : (tensor<3x42x76x45xi16>) -> tensor<3x1x76x45xi16>
    %1 = tosa.bitwise_or %0, %0 : (tensor<3x1x76x45xi16>, tensor<3x1x76x45xi16>) -> tensor<3x1x76x45xi16>
    %2 = tosa.equal %arg1, %arg2 : (tensor<i8>, tensor<i8>) -> tensor<i1>
    %3 = tosa.logical_xor %2, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.bitwise_not %3 : (tensor<i1>) -> tensor<i1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %6 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<3x1x76x45xi16>) -> tensor<1x1x76x45xi16>
    %7 = tosa.logical_not %5 : (tensor<i1>) -> tensor<i1>
    %8 = tosa.sub %2, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %9 = tosa.bitwise_and %8, %7 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %in_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %10 = tosa.negate %7, %in_zp_10, %out_zp_10 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %11 = tosa.logical_and %9, %8 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %12 = tosa.ceil %arg3 : (tensor<33x31x79xf32>) -> tensor<33x31x79xf32>
    %13 = tosa.sigmoid %12 : (tensor<33x31x79xf32>) -> tensor<33x31x79xf32>
    %14 = tosa.reverse %12 {axis = 2 : i32} : (tensor<33x31x79xf32>) -> tensor<33x31x79xf32>
    %15 = tosa.argmax %13 {axis = 2 : i32} : (tensor<33x31x79xf32>) -> tensor<33x31xi32>
    %16 = tosa.arithmetic_right_shift %15, %15 {round = false} : (tensor<33x31xi32>, tensor<33x31xi32>) -> tensor<33x31xi32>
    %17 = tosa.clamp %16 {min_val = 0 : i32, max_val = 124 : i32} : (tensor<33x31xi32>) -> tensor<33x31xi32>
    %18 = tosa.logical_right_shift %17, %16 : (tensor<33x31xi32>, tensor<33x31xi32>) -> tensor<33x31xi32>
    %19 = tosa.concat %12, %14 {axis = 0 : i32} : (tensor<33x31x79xf32>, tensor<33x31x79xf32>) -> tensor<66x31x79xf32>
    %20 = tosa.logical_left_shift %18, %15 : (tensor<33x31xi32>, tensor<33x31xi32>) -> tensor<33x31xi32>
    return %1, %6, %10, %11, %19, %20 : tensor<3x1x76x45xi16>, tensor<1x1x76x45xi16>, tensor<i1>, tensor<i1>, tensor<66x31x79xf32>, tensor<33x31xi32>
  }
}
