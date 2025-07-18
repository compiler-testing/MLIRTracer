module {
  func.func @main(%arg0: tensor<81xi16>, %arg1: tensor<50x27x34x87xi1>, %arg2: tensor<1x27x1x87xi1>, %arg3: tensor<12x45x63x1x50x63xf32>, %arg4: tensor<12x1x1x1x50x1xf32>) -> (tensor<1xi16>, tensor<50x27x34x87xi1>, tensor<12x45x63x1x50x63xf32>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<81xi16>) -> tensor<1xi16>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<50x27x34x87xi1>, tensor<1x27x1x87xi1>) -> tensor<50x27x34x87xi1>
    %2 = tosa.logical_not %1 : (tensor<50x27x34x87xi1>) -> tensor<50x27x34x87xi1>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<50x27x34x87xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<50x27x34x87xi1>
    %4 = tosa.sub %2, %3 : (tensor<50x27x34x87xi1>, tensor<50x27x34x87xi1>) -> tensor<50x27x34x87xi1>
    %5 = tosa.minimum %arg3, %arg4 : (tensor<12x45x63x1x50x63xf32>, tensor<12x1x1x1x50x1xf32>) -> tensor<12x45x63x1x50x63xf32>
    return %0, %4, %5 : tensor<1xi16>, tensor<50x27x34x87xi1>, tensor<12x45x63x1x50x63xf32>
  }
}
