module {
  func.func @main(%arg0: tensor<63x8x64x83xi16>, %arg1: tensor<41x85x66xi1>, %arg2: tensor<1x85x66xi1>, %arg3: tensor<70x70x45x55x37xi32>, %arg4: tensor<1x1x1x55x37xi32>, %arg5: tensor<26x49x96x58xf32>) -> (tensor<1x1x10x1xi16>, tensor<70x70x45x55x37xi1>, tensor<1x6x6xi1>, tensor<11x6x6xi1>, tensor<26x49x96x58xf32>) {
    %0 = tosa.reduce_product %arg0 {axis = 1 : i32} : (tensor<63x8x64x83xi16>) -> tensor<63x1x64x83xi16>
    %s_1_start = tosa.const_shape {values = dense<[ 29, 0, 2, 18 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_1_size = tosa.const_shape {values = dense<[ 10, 8, 10, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<63x1x64x83xi16>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<10x8x10x1xi16>
    %2 = tosa.logical_or %arg1, %arg2 : (tensor<41x85x66xi1>, tensor<1x85x66xi1>) -> tensor<41x85x66xi1>
    %3 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<10x8x10x1xi16>) -> tensor<10x1x10x1xi16>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<10x1x10x1xi16>) -> tensor<1x1x10x1xi16>
    %5 = tosa.reverse %4 {axis = 0 : i32} : (tensor<1x1x10x1xi16>) -> tensor<1x1x10x1xi16>
    %6 = tosa.abs %5 : (tensor<1x1x10x1xi16>) -> tensor<1x1x10x1xi16>
    %s_7_start = tosa.const_shape {values = dense<[ 30, 29, 22 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_7_size = tosa.const_shape {values = dense<[ 11, 6, 6 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %7 = tosa.slice %2, %s_7_start, %s_7_size : (tensor<41x85x66xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<11x6x6xi1>
    %8 = tosa.greater_equal %arg3, %arg4 : (tensor<70x70x45x55x37xi32>, tensor<1x1x1x55x37xi32>) -> tensor<70x70x45x55x37xi1>
    %9 = tosa.sub %6, %6 : (tensor<1x1x10x1xi16>, tensor<1x1x10x1xi16>) -> tensor<1x1x10x1xi16>
    %10 = tosa.logical_or %8, %8 : (tensor<70x70x45x55x37xi1>, tensor<70x70x45x55x37xi1>) -> tensor<70x70x45x55x37xi1>
    %11 = tosa.reverse %7 {axis = 0 : i32} : (tensor<11x6x6xi1>) -> tensor<11x6x6xi1>
    %12 = tosa.sub %11, %7 : (tensor<11x6x6xi1>, tensor<11x6x6xi1>) -> tensor<11x6x6xi1>
    %13 = tosa.reduce_sum %12 {axis = 0 : i32} : (tensor<11x6x6xi1>) -> tensor<1x6x6xi1>
    %14 = tosa.logical_left_shift %7, %12 : (tensor<11x6x6xi1>, tensor<11x6x6xi1>) -> tensor<11x6x6xi1>
    %15 = tosa.exp %arg5 : (tensor<26x49x96x58xf32>) -> tensor<26x49x96x58xf32>
    return %9, %10, %13, %14, %15 : tensor<1x1x10x1xi16>, tensor<70x70x45x55x37xi1>, tensor<1x6x6xi1>, tensor<11x6x6xi1>, tensor<26x49x96x58xf32>
  }
}
