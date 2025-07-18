module {
  func.func @main(%arg0: tensor<75x51x6xi1>, %arg1: tensor<34x64x25x74xf32>) -> (tensor<102x128x50x148xf32>, tensor<1x51x6xi1>, tensor<102x128x50x1xf32>, tensor<1x51x1xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<75x51x6xi1>) -> tensor<1x51x6xi1>
    %1 = tosa.reduce_all %0 {axis = 2 : i32} : (tensor<1x51x6xi1>) -> tensor<1x51x1xi1>
    %2 = tosa.log %arg1 : (tensor<34x64x25x74xf32>) -> tensor<34x64x25x74xf32>
    %t_3 = tosa.const_shape {values = dense<[ 3, 2, 2, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.tile %2, %t_3 : (tensor<34x64x25x74xf32>, !tosa.shape<4>) -> tensor<102x128x50x148xf32>
    %4 = tosa.rsqrt %3 : (tensor<102x128x50x148xf32>) -> tensor<102x128x50x148xf32>
    %5 = tosa.reduce_min %3 {axis = 3 : i32} : (tensor<102x128x50x148xf32>) -> tensor<102x128x50x1xf32>
    %6 = tosa.logical_left_shift %0, %0 : (tensor<1x51x6xi1>, tensor<1x51x6xi1>) -> tensor<1x51x6xi1>
    %7 = tosa.sub %5, %5 : (tensor<102x128x50x1xf32>, tensor<102x128x50x1xf32>) -> tensor<102x128x50x1xf32>
    %8 = tosa.logical_or %1, %1 : (tensor<1x51x1xi1>, tensor<1x51x1xi1>) -> tensor<1x51x1xi1>
    return %4, %6, %7, %8 : tensor<102x128x50x148xf32>, tensor<1x51x6xi1>, tensor<102x128x50x1xf32>, tensor<1x51x1xi1>
  }
}
