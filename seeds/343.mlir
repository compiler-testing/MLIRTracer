module {
  func.func @main(%arg0: tensor<26x30x18x65x58x31xf32>, %arg1: tensor<1x1x18x65x58x31xf32>, %arg2: tensor<14x6x95x36xf32>, %arg3: tensor<42x59x60x62xi32>, %arg4: tensor<42x1x60x62xi32>) -> (tensor<26x30x18x65x58x31xi1>, tensor<42x118x60x62xi32>, tensor<14x6x95x36xf32>, tensor<14x6x95x36xf32>, tensor<3x2x2x6xi32>, tensor<26x30x18x65x58x31xi1>, tensor<42x59x1x62xi32>, tensor<14x6x95x36xf32>, tensor<14x6x95x36xf32>, tensor<42x59x60x62xi1>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<26x30x18x65x58x31xf32>, tensor<1x1x18x65x58x31xf32>) -> tensor<26x30x18x65x58x31xi1>
    %1 = tosa.logical_xor %0, %0 : (tensor<26x30x18x65x58x31xi1>, tensor<26x30x18x65x58x31xi1>) -> tensor<26x30x18x65x58x31xi1>
    %2 = tosa.sub %1, %0 : (tensor<26x30x18x65x58x31xi1>, tensor<26x30x18x65x58x31xi1>) -> tensor<26x30x18x65x58x31xi1>
    %3 = tosa.ceil %arg2 : (tensor<14x6x95x36xf32>) -> tensor<14x6x95x36xf32>
    %4 = tosa.sigmoid %3 : (tensor<14x6x95x36xf32>) -> tensor<14x6x95x36xf32>
    %5 = tosa.intdiv %arg3, %arg4 : (tensor<42x59x60x62xi32>, tensor<42x1x60x62xi32>) -> tensor<42x59x60x62xi32>
    %6 = tosa.concat %5, %5 {axis = 1 : i32} : (tensor<42x59x60x62xi32>, tensor<42x59x60x62xi32>) -> tensor<42x118x60x62xi32>
    %7 = tosa.bitwise_or %5, %5 : (tensor<42x59x60x62xi32>, tensor<42x59x60x62xi32>) -> tensor<42x59x60x62xi32>
    %8 = tosa.log %4 : (tensor<14x6x95x36xf32>) -> tensor<14x6x95x36xf32>
    %9 = tosa.reverse %7 {axis = 3 : i32} : (tensor<42x59x60x62xi32>) -> tensor<42x59x60x62xi32>
    %10 = tosa.maximum %5, %9 : (tensor<42x59x60x62xi32>, tensor<42x59x60x62xi32>) -> tensor<42x59x60x62xi32>
    %11 = tosa.log %3 : (tensor<14x6x95x36xf32>) -> tensor<14x6x95x36xf32>
    %12 = tosa.ceil %8 : (tensor<14x6x95x36xf32>) -> tensor<14x6x95x36xf32>
    %13 = tosa.pow %3, %11 : (tensor<14x6x95x36xf32>, tensor<14x6x95x36xf32>) -> tensor<14x6x95x36xf32>
    %14 = tosa.intdiv %9, %7 : (tensor<42x59x60x62xi32>, tensor<42x59x60x62xi32>) -> tensor<42x59x60x62xi32>
    %s_15_start = tosa.const_shape {values = dense<[ 33, 25, 39, 34 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_15_size = tosa.const_shape {values = dense<[ 3, 2, 2, 6 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %15 = tosa.slice %10, %s_15_start, %s_15_size : (tensor<42x59x60x62xi32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x2x2x6xi32>
    %16 = tosa.logical_and %0, %1 : (tensor<26x30x18x65x58x31xi1>, tensor<26x30x18x65x58x31xi1>) -> tensor<26x30x18x65x58x31xi1>
    %17 = tosa.reduce_min %10 {axis = 2 : i32} : (tensor<42x59x60x62xi32>) -> tensor<42x59x1x62xi32>
    %18 = tosa.bitwise_and %14, %10 : (tensor<42x59x60x62xi32>, tensor<42x59x60x62xi32>) -> tensor<42x59x60x62xi32>
    %19 = tosa.pow %4, %11 : (tensor<14x6x95x36xf32>, tensor<14x6x95x36xf32>) -> tensor<14x6x95x36xf32>
    %20 = tosa.exp %8 : (tensor<14x6x95x36xf32>) -> tensor<14x6x95x36xf32>
    %21 = tosa.greater %18, %10 : (tensor<42x59x60x62xi32>, tensor<42x59x60x62xi32>) -> tensor<42x59x60x62xi1>
    return %2, %6, %12, %13, %15, %16, %17, %19, %20, %21 : tensor<26x30x18x65x58x31xi1>, tensor<42x118x60x62xi32>, tensor<14x6x95x36xf32>, tensor<14x6x95x36xf32>, tensor<3x2x2x6xi32>, tensor<26x30x18x65x58x31xi1>, tensor<42x59x1x62xi32>, tensor<14x6x95x36xf32>, tensor<14x6x95x36xf32>, tensor<42x59x60x62xi1>
  }
}
