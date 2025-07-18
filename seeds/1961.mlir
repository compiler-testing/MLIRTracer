module {
  func.func @main(%arg0: tensor<38x40x83x6xi16>, %arg1: tensor<38x40x83x1xi16>, %arg2: tensor<6x5x45x64x53xi1>, %arg3: tensor<6x5x1x64x53xi1>, %arg4: tensor<34x11x8x14xf32>) -> (tensor<6x5x45x64x53xi1>, tensor<38x40x2x18xi16>, tensor<6x8x12x10xf32>, tensor<34x1x8x14xf32>, tensor<34x3x24x42xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<38x40x83x6xi16>, tensor<38x40x83x1xi16>) -> tensor<38x40x83x6xi16>
    %1 = tosa.clamp %0 {min_val = 4 : i16, max_val = 82 : i16} : (tensor<38x40x83x6xi16>) -> tensor<38x40x83x6xi16>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<6x5x45x64x53xi1>, tensor<6x5x1x64x53xi1>) -> tensor<6x5x45x64x53xi1>
    %3 = tosa.log %arg4 : (tensor<34x11x8x14xf32>) -> tensor<34x11x8x14xf32>
    %4 = tosa.reduce_min %3 {axis = 1 : i32} : (tensor<34x11x8x14xf32>) -> tensor<34x1x8x14xf32>
    %5 = tosa.reduce_min %1 {axis = 2 : i32} : (tensor<38x40x83x6xi16>) -> tensor<38x40x1x6xi16>
    %t_6 = tosa.const_shape {values = dense<[ 1, 1, 2, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %6 = tosa.tile %5, %t_6 : (tensor<38x40x1x6xi16>, !tosa.shape<4>) -> tensor<38x40x2x18xi16>
    %7 = tosa.bitwise_not %6 : (tensor<38x40x2x18xi16>) -> tensor<38x40x2x18xi16>
    %s_8_start = tosa.const_shape {values = dense<[ 18, 0, 0, 4 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_8_size = tosa.const_shape {values = dense<[ 6, 8, 12, 10 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %8 = tosa.slice %4, %s_8_start, %s_8_size : (tensor<34x1x8x14xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<6x8x12x10xf32>
    %t_9 = tosa.const_shape {values = dense<[ 1, 3, 3, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %9 = tosa.tile %4, %t_9 : (tensor<34x1x8x14xf32>, !tosa.shape<4>) -> tensor<34x3x24x42xf32>
    %10 = tosa.maximum %4, %4 : (tensor<34x1x8x14xf32>, tensor<34x1x8x14xf32>) -> tensor<34x1x8x14xf32>
    %11 = tosa.add %9, %9 : (tensor<34x3x24x42xf32>, tensor<34x3x24x42xf32>) -> tensor<34x3x24x42xf32>
    return %2, %7, %8, %10, %11 : tensor<6x5x45x64x53xi1>, tensor<38x40x2x18xi16>, tensor<6x8x12x10xf32>, tensor<34x1x8x14xf32>, tensor<34x3x24x42xf32>
  }
}
