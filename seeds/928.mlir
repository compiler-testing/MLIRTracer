module {
  func.func @main(%arg0: tensor<24x22xf32>, %arg1: tensor<1x22xf32>, %arg2: tensor<20x15x27x14x33xi32>, %arg3: tensor<1x1x1x14x33xi32>, %arg4: tensor<94x90x45x7x2x85xi1>, %arg5: tensor<1x90x45x1x2x85xi1>, %arg6: tensor<9xi1>) -> (tensor<94x90x45x7x2x85xi1>, tensor<2x4x3x10x12xi32>, tensor<24x22xf32>, tensor<1xi1>, tensor<1xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<24x22xf32>, tensor<1x22xf32>) -> tensor<24x22xf32>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<24x22xf32>) -> tensor<24x22xf32>
    %2 = tosa.sigmoid %1 : (tensor<24x22xf32>) -> tensor<24x22xf32>
    %3 = tosa.intdiv %arg2, %arg3 : (tensor<20x15x27x14x33xi32>, tensor<1x1x1x14x33xi32>) -> tensor<20x15x27x14x33xi32>
    %4 = tosa.maximum %2, %1 : (tensor<24x22xf32>, tensor<24x22xf32>) -> tensor<24x22xf32>
    %5 = tosa.logical_or %arg4, %arg5 : (tensor<94x90x45x7x2x85xi1>, tensor<1x90x45x1x2x85xi1>) -> tensor<94x90x45x7x2x85xi1>
    %6 = tosa.maximum %4, %1 : (tensor<24x22xf32>, tensor<24x22xf32>) -> tensor<24x22xf32>
    %7 = tosa.reverse %6 {axis = 1 : i32} : (tensor<24x22xf32>) -> tensor<24x22xf32>
    %8 = tosa.reduce_any %arg6 {axis = 0 : i32} : (tensor<9xi1>) -> tensor<1xi1>
    %s_9_start = tosa.const_shape {values = dense<[ 8, 11, 13, 4, 1 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_9_size = tosa.const_shape {values = dense<[ 2, 4, 3, 10, 12 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %9 = tosa.slice %3, %s_9_start, %s_9_size : (tensor<20x15x27x14x33xi32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<2x4x3x10x12xi32>
    %10 = tosa.sub %7, %2 : (tensor<24x22xf32>, tensor<24x22xf32>) -> tensor<24x22xf32>
    %11 = tosa.logical_not %8 : (tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.reduce_min %8 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %5, %9, %10, %11, %12 : tensor<94x90x45x7x2x85xi1>, tensor<2x4x3x10x12xi32>, tensor<24x22xf32>, tensor<1xi1>, tensor<1xi1>
  }
}
