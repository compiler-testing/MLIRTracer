module {
  func.func @main(%arg0: tensor<89x58xi16>, %arg1: tensor<89x1xi16>, %arg2: tensor<21x94xi1>, %arg3: tensor<21x1xi1>, %arg4: tensor<53x39x64x49xf32>) -> (tensor<1xi16>, tensor<53x39x64x49xi1>, tensor<1x94xi1>, tensor<1x94xi1>, tensor<53x39x64x49xi1>, tensor<1x94xi1>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<89x58xi16>, tensor<89x1xi16>) -> tensor<89x58xi16>
    %r_1 = tosa.const_shape {values = dense<[ 5162 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.reshape %0, %r_1 : (tensor<89x58xi16>, !tosa.shape<1>) -> tensor<5162xi16>
    %2 = tosa.abs %1 : (tensor<5162xi16>) -> tensor<5162xi16>
    %3 = tosa.concat %2, %2 {axis = 0 : i32} : (tensor<5162xi16>, tensor<5162xi16>) -> tensor<10324xi16>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %4 = tosa.negate %3, %in_zp_4, %out_zp_4 : (tensor<10324xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<10324xi16>
    %5 = tosa.reduce_sum %4 {axis = 0 : i32} : (tensor<10324xi16>) -> tensor<1xi16>
    %6 = tosa.logical_and %arg2, %arg3 : (tensor<21x94xi1>, tensor<21x1xi1>) -> tensor<21x94xi1>
    %7 = tosa.floor %arg4 : (tensor<53x39x64x49xf32>) -> tensor<53x39x64x49xf32>
    %8 = tosa.reduce_max %6 {axis = 0 : i32} : (tensor<21x94xi1>) -> tensor<1x94xi1>
    %9 = tosa.arithmetic_right_shift %8, %8 {round = true} : (tensor<1x94xi1>, tensor<1x94xi1>) -> tensor<1x94xi1>
    %10 = tosa.greater_equal %7, %7 : (tensor<53x39x64x49xf32>, tensor<53x39x64x49xf32>) -> tensor<53x39x64x49xi1>
    %11 = tosa.logical_and %9, %9 : (tensor<1x94xi1>, tensor<1x94xi1>) -> tensor<1x94xi1>
    %12 = tosa.clz %8 : (tensor<1x94xi1>) -> tensor<1x94xi1>
    %13 = tosa.logical_and %8, %12 : (tensor<1x94xi1>, tensor<1x94xi1>) -> tensor<1x94xi1>
    %14 = tosa.greater %7, %7 : (tensor<53x39x64x49xf32>, tensor<53x39x64x49xf32>) -> tensor<53x39x64x49xi1>
    %15 = tosa.reduce_any %8 {axis = 0 : i32} : (tensor<1x94xi1>) -> tensor<1x94xi1>
    return %5, %10, %11, %13, %14, %15 : tensor<1xi16>, tensor<53x39x64x49xi1>, tensor<1x94xi1>, tensor<1x94xi1>, tensor<53x39x64x49xi1>, tensor<1x94xi1>
  }
}
