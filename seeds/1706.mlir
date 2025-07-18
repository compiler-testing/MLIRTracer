module {
  func.func @main(%arg0: tensor<46x55xi16>, %arg1: tensor<1x1xi16>, %arg2: tensor<95x21x64x94xf32>) -> (tensor<46x1xi16>, tensor<95x21x64x94xi1>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<46x55xi16>, tensor<1x1xi16>) -> tensor<46x55xi16>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<46x55xi16>) -> tensor<46x1xi16>
    %2 = tosa.exp %arg2 : (tensor<95x21x64x94xf32>) -> tensor<95x21x64x94xf32>
    %3 = tosa.reverse %2 {axis = 0 : i32} : (tensor<95x21x64x94xf32>) -> tensor<95x21x64x94xf32>
    %4 = tosa.abs %1 : (tensor<46x1xi16>) -> tensor<46x1xi16>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<46x1xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<46x1xi16>
    %6 = tosa.clamp %5 {min_val = -21 : i16, max_val = 75 : i16} : (tensor<46x1xi16>) -> tensor<46x1xi16>
    %7 = tosa.greater_equal %3, %3 : (tensor<95x21x64x94xf32>, tensor<95x21x64x94xf32>) -> tensor<95x21x64x94xi1>
    return %6, %7 : tensor<46x1xi16>, tensor<95x21x64x94xi1>
  }
}
