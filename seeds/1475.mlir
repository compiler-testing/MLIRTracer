module {
  func.func @main(%arg0: tensor<32x63x51x92x68x70xf32>, %arg1: tensor<33x57x48x22x70x69xi1>, %arg2: tensor<1x1x1x22x70x69xi1>) -> (tensor<2x9x5x1x3x4xf32>, tensor<3x1x9x5x2x4xf32>, tensor<3x1x9x5x2x4xi1>, tensor<33x57x48x22x70x69xi1>, tensor<3x1x9x5x2x4xf32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 20, 13, 3, 24, 30, 13 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_0_size = tosa.const_shape {values = dense<[ 2, 9, 5, 1, 3, 4 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<32x63x51x92x68x70xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<2x9x5x1x3x4xf32>
    %1 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<2x9x5x1x3x4xf32>) -> tensor<3x1x9x5x2x4xf32>
    %3 = tosa.floor %2 : (tensor<3x1x9x5x2x4xf32>) -> tensor<3x1x9x5x2x4xf32>
    %4 = tosa.logical_and %arg1, %arg2 : (tensor<33x57x48x22x70x69xi1>, tensor<1x1x1x22x70x69xi1>) -> tensor<33x57x48x22x70x69xi1>
    %5 = tosa.rsqrt %3 : (tensor<3x1x9x5x2x4xf32>) -> tensor<3x1x9x5x2x4xf32>
    %6 = tosa.logical_xor %4, %4 : (tensor<33x57x48x22x70x69xi1>, tensor<33x57x48x22x70x69xi1>) -> tensor<33x57x48x22x70x69xi1>
    %7 = tosa.tanh %0 : (tensor<2x9x5x1x3x4xf32>) -> tensor<2x9x5x1x3x4xf32>
    %8 = tosa.exp %5 : (tensor<3x1x9x5x2x4xf32>) -> tensor<3x1x9x5x2x4xf32>
    %9 = tosa.greater_equal %5, %3 : (tensor<3x1x9x5x2x4xf32>, tensor<3x1x9x5x2x4xf32>) -> tensor<3x1x9x5x2x4xi1>
    %in_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %10 = tosa.negate %6, %in_zp_10, %out_zp_10 : (tensor<33x57x48x22x70x69xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<33x57x48x22x70x69xi1>
    %11 = tosa.sigmoid %5 : (tensor<3x1x9x5x2x4xf32>) -> tensor<3x1x9x5x2x4xf32>
    return %7, %8, %9, %10, %11 : tensor<2x9x5x1x3x4xf32>, tensor<3x1x9x5x2x4xf32>, tensor<3x1x9x5x2x4xi1>, tensor<33x57x48x22x70x69xi1>, tensor<3x1x9x5x2x4xf32>
  }
}
