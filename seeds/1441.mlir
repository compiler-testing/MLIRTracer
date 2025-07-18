module {
  func.func @main(%arg0: tensor<3x4x98x35x39x39xi16>, %arg1: tensor<61x92x19x60x21xf32>, %arg2: tensor<61x1x1x1x21xf32>, %arg3: tensor<98x93x12x43xi1>) -> (tensor<3x4x98x35x39x39xi16>, tensor<61x92x19x60x21xi1>, tensor<294x186x36x1xi1>, tensor<61x92x19x60x21xf32>, tensor<98x1x12x1xi1>, tensor<61x92x19x60x21xi1>) {
    %0 = tosa.abs %arg0 : (tensor<3x4x98x35x39x39xi16>) -> tensor<3x4x98x35x39x39xi16>
    %1 = tosa.pow %arg1, %arg2 : (tensor<61x92x19x60x21xf32>, tensor<61x1x1x1x21xf32>) -> tensor<61x92x19x60x21xf32>
    %2 = tosa.sigmoid %1 : (tensor<61x92x19x60x21xf32>) -> tensor<61x92x19x60x21xf32>
    %3 = tosa.sigmoid %2 : (tensor<61x92x19x60x21xf32>) -> tensor<61x92x19x60x21xf32>
    %4 = tosa.bitwise_and %0, %0 : (tensor<3x4x98x35x39x39xi16>, tensor<3x4x98x35x39x39xi16>) -> tensor<3x4x98x35x39x39xi16>
    %5 = tosa.bitwise_and %4, %4 : (tensor<3x4x98x35x39x39xi16>, tensor<3x4x98x35x39x39xi16>) -> tensor<3x4x98x35x39x39xi16>
    %6 = tosa.equal %3, %2 : (tensor<61x92x19x60x21xf32>, tensor<61x92x19x60x21xf32>) -> tensor<61x92x19x60x21xi1>
    %7 = tosa.logical_left_shift %6, %6 : (tensor<61x92x19x60x21xi1>, tensor<61x92x19x60x21xi1>) -> tensor<61x92x19x60x21xi1>
    %8 = tosa.add %7, %6 : (tensor<61x92x19x60x21xi1>, tensor<61x92x19x60x21xi1>) -> tensor<61x92x19x60x21xi1>
    %9 = tosa.logical_and %8, %6 : (tensor<61x92x19x60x21xi1>, tensor<61x92x19x60x21xi1>) -> tensor<61x92x19x60x21xi1>
    %10 = tosa.reduce_product %arg3 {axis = 3 : i32} : (tensor<98x93x12x43xi1>) -> tensor<98x93x12x1xi1>
    %t_11 = tosa.const_shape {values = dense<[ 3, 2, 3, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %11 = tosa.tile %10, %t_11 : (tensor<98x93x12x1xi1>, !tosa.shape<4>) -> tensor<294x186x36x1xi1>
    %12 = tosa.log %1 : (tensor<61x92x19x60x21xf32>) -> tensor<61x92x19x60x21xf32>
    %13 = tosa.reduce_product %10 {axis = 1 : i32} : (tensor<98x93x12x1xi1>) -> tensor<98x1x12x1xi1>
    %14 = tosa.greater_equal %1, %2 : (tensor<61x92x19x60x21xf32>, tensor<61x92x19x60x21xf32>) -> tensor<61x92x19x60x21xi1>
    return %5, %9, %11, %12, %13, %14 : tensor<3x4x98x35x39x39xi16>, tensor<61x92x19x60x21xi1>, tensor<294x186x36x1xi1>, tensor<61x92x19x60x21xf32>, tensor<98x1x12x1xi1>, tensor<61x92x19x60x21xi1>
  }
}
