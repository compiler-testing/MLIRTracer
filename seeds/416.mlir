module {
  func.func @main(%arg0: tensor<83x45x58x82x2x69xi16>, %arg1: tensor<6x2xi64>, %arg2: tensor<58x31x28x16xf32>) -> (tensor<12x11x12x12x8x3xi16>, tensor<83x45x58x82x2x69xi16>, tensor<83x45x58x82x2x69xi16>, tensor<58x31x28x16xf32>, tensor<58x1x28x16xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<12xindex>} : () -> !tosa.shape<12>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<83x45x58x82x2x69xi16>, !tosa.shape<12>, tensor<1xi16>) -> tensor<83x45x58x82x2x69xi16>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<83x45x58x82x2x69xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<83x45x58x82x2x69xi16>
    %s_2_start = tosa.const_shape {values = dense<[ 44, 4, 44, 56, 0, 31 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_2_size = tosa.const_shape {values = dense<[ 12, 11, 12, 12, 8, 3 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<83x45x58x82x2x69xi16>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<12x11x12x12x8x3xi16>
    %3 = tosa.reverse %arg2 {axis = 0 : i32} : (tensor<58x31x28x16xf32>) -> tensor<58x31x28x16xf32>
    %4 = tosa.reduce_sum %3 {axis = 1 : i32} : (tensor<58x31x28x16xf32>) -> tensor<58x1x28x16xf32>
    %5 = tosa.bitwise_not %0 : (tensor<83x45x58x82x2x69xi16>) -> tensor<83x45x58x82x2x69xi16>
    %6 = tosa.exp %3 : (tensor<58x31x28x16xf32>) -> tensor<58x31x28x16xf32>
    %7 = tosa.pow %6, %6 : (tensor<58x31x28x16xf32>, tensor<58x31x28x16xf32>) -> tensor<58x31x28x16xf32>
    %8 = tosa.clz %1 : (tensor<83x45x58x82x2x69xi16>) -> tensor<83x45x58x82x2x69xi16>
    %9 = tosa.abs %4 : (tensor<58x1x28x16xf32>) -> tensor<58x1x28x16xf32>
    %10 = tosa.rsqrt %7 : (tensor<58x31x28x16xf32>) -> tensor<58x31x28x16xf32>
    %11 = tosa.abs %9 : (tensor<58x1x28x16xf32>) -> tensor<58x1x28x16xf32>
    return %2, %5, %8, %10, %11 : tensor<12x11x12x12x8x3xi16>, tensor<83x45x58x82x2x69xi16>, tensor<83x45x58x82x2x69xi16>, tensor<58x31x28x16xf32>, tensor<58x1x28x16xf32>
  }
}
