module {
  func.func @main(%arg0: tensor<61x30x63x21x5x83xi8>, %arg1: tensor<10x18xi1>, %arg2: tensor<49xf32>) -> (tensor<61x30x63x21x5x83xi8>, tensor<49xf32>, tensor<1x1xi1>, tensor<10x1xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<61x30x63x21x5x83xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<61x30x63x21x5x83xi8>
    %1 = tosa.abs %0 : (tensor<61x30x63x21x5x83xi8>) -> tensor<61x30x63x21x5x83xi8>
    %2 = tosa.reduce_sum %arg1 {axis = 1 : i32} : (tensor<10x18xi1>) -> tensor<10x1xi1>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<61x30x63x21x5x83xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<61x30x63x21x5x83xi8>
    %4 = tosa.minimum %3, %1 : (tensor<61x30x63x21x5x83xi8>, tensor<61x30x63x21x5x83xi8>) -> tensor<61x30x63x21x5x83xi8>
    %5 = tosa.reverse %2 {axis = 0 : i32} : (tensor<10x1xi1>) -> tensor<10x1xi1>
    %6 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<10x1xi1>) -> tensor<1x1xi1>
    %7 = tosa.rsqrt %arg2 : (tensor<49xf32>) -> tensor<49xf32>
    %8 = tosa.reduce_min %6 {axis = 0 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %9 = tosa.bitwise_not %5 : (tensor<10x1xi1>) -> tensor<10x1xi1>
    return %4, %7, %8, %9 : tensor<61x30x63x21x5x83xi8>, tensor<49xf32>, tensor<1x1xi1>, tensor<10x1xi1>
  }
}
