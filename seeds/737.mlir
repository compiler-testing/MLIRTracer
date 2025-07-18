module {
  func.func @main(%arg0: tensor<16x17xi8>, %arg1: tensor<23x50x40x18x51xf32>, %arg2: tensor<63x94xi1>) -> (tensor<2x51xi8>, tensor<23x50x40x18x51xf32>, tensor<63x94xi1>, tensor<1x94xi1>, tensor<23x50x40x18x51xf32>) {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<16x17xi8>) -> tensor<1x17xi8>
    %1 = tosa.exp %arg1 : (tensor<23x50x40x18x51xf32>) -> tensor<23x50x40x18x51xf32>
    %2 = tosa.logical_not %arg2 : (tensor<63x94xi1>) -> tensor<63x94xi1>
    %3 = tosa.reverse %0 {axis = 1 : i32} : (tensor<1x17xi8>) -> tensor<1x17xi8>
    %t_4 = tosa.const_shape {values = dense<[ 2, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.tile %3, %t_4 : (tensor<1x17xi8>, !tosa.shape<2>) -> tensor<2x51xi8>
    %5 = tosa.minimum %1, %1 : (tensor<23x50x40x18x51xf32>, tensor<23x50x40x18x51xf32>) -> tensor<23x50x40x18x51xf32>
    %6 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<63x94xi1>) -> tensor<1x94xi1>
    %7 = tosa.logical_not %2 : (tensor<63x94xi1>) -> tensor<63x94xi1>
    %8 = tosa.tanh %1 : (tensor<23x50x40x18x51xf32>) -> tensor<23x50x40x18x51xf32>
    %9 = tosa.abs %7 : (tensor<63x94xi1>) -> tensor<63x94xi1>
    %t_10 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %10 = tosa.tile %6, %t_10 : (tensor<1x94xi1>, !tosa.shape<2>) -> tensor<1x94xi1>
    %11 = tosa.logical_left_shift %6, %10 : (tensor<1x94xi1>, tensor<1x94xi1>) -> tensor<1x94xi1>
    %12 = tosa.logical_left_shift %11, %6 : (tensor<1x94xi1>, tensor<1x94xi1>) -> tensor<1x94xi1>
    %13 = tosa.exp %8 : (tensor<23x50x40x18x51xf32>) -> tensor<23x50x40x18x51xf32>
    return %4, %5, %9, %12, %13 : tensor<2x51xi8>, tensor<23x50x40x18x51xf32>, tensor<63x94xi1>, tensor<1x94xi1>, tensor<23x50x40x18x51xf32>
  }
}
