module {
  func.func @main(%arg0: tensor<1x74x71xi32>, %arg1: tensor<1x74x71xi32>, %arg2: tensor<59x67x12x50xf32>, %arg3: tensor<78x89x24x19x82x25xi1>, %arg4: tensor<78x1x1x19x82x1xi1>, %arg5: tensor<94x59x69x28xi1>) -> (tensor<1x1x1xi32>, tensor<59x1x1x1xf32>, tensor<2x1x7x9x10x9xi1>, tensor<59x1x24x1xf32>, tensor<826x1x1xi1>, tensor<1x59x1x28xi1>, tensor<1x59x1x28xi1>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<1x74x71xi32>, tensor<1x74x71xi32>) -> tensor<1x74x71xi32>
    %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x74x71xi32>) -> tensor<1x1x71xi32>
    %2 = tosa.exp %arg2 : (tensor<59x67x12x50xf32>) -> tensor<59x67x12x50xf32>
    %3 = tosa.concat %2, %2 {axis = 2 : i32} : (tensor<59x67x12x50xf32>, tensor<59x67x12x50xf32>) -> tensor<59x67x24x50xf32>
    %4 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<1x1x71xi32>) -> tensor<1x1x71xi32>
    %5 = tosa.reduce_min %3 {axis = 3 : i32} : (tensor<59x67x24x50xf32>) -> tensor<59x67x24x1xf32>
    %6 = tosa.bitwise_or %4, %1 : (tensor<1x1x71xi32>, tensor<1x1x71xi32>) -> tensor<1x1x71xi32>
    %7 = tosa.reduce_min %5 {axis = 1 : i32} : (tensor<59x67x24x1xf32>) -> tensor<59x1x24x1xf32>
    %8 = tosa.reduce_product %6 {axis = 2 : i32} : (tensor<1x1x71xi32>) -> tensor<1x1x1xi32>
    %9 = tosa.logical_or %arg3, %arg4 : (tensor<78x89x24x19x82x25xi1>, tensor<78x1x1x19x82x1xi1>) -> tensor<78x89x24x19x82x25xi1>
    %10 = tosa.reverse %8 {axis = 2 : i32} : (tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
    %11 = tosa.reduce_sum %10 {axis = 2 : i32} : (tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
    %12 = tosa.reciprocal %7 : (tensor<59x1x24x1xf32>) -> tensor<59x1x24x1xf32>
    %13 = tosa.reduce_product %7 {axis = 2 : i32} : (tensor<59x1x24x1xf32>) -> tensor<59x1x1x1xf32>
    %s_14_start = tosa.const_shape {values = dense<[ 25, 8, 8, 10, 47, 16 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_14_size = tosa.const_shape {values = dense<[ 2, 1, 7, 9, 10, 9 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %14 = tosa.slice %9, %s_14_start, %s_14_size : (tensor<78x89x24x19x82x25xi1>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<2x1x7x9x10x9xi1>
    %15 = tosa.reduce_any %arg5 {axis = 0 : i32} : (tensor<94x59x69x28xi1>) -> tensor<1x59x69x28xi1>
    %16 = tosa.reduce_max %15 {axis = 2 : i32} : (tensor<1x59x69x28xi1>) -> tensor<1x59x1x28xi1>
    %17 = tosa.log %12 : (tensor<59x1x24x1xf32>) -> tensor<59x1x24x1xf32>
    %r_18 = tosa.const_shape {values = dense<[ 826, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %18 = tosa.reshape %16, %r_18 : (tensor<1x59x1x28xi1>, !tosa.shape<3>) -> tensor<826x1x2xi1>
    %19 = tosa.reduce_all %18 {axis = 2 : i32} : (tensor<826x1x2xi1>) -> tensor<826x1x1xi1>
    %20 = tosa.bitwise_not %16 : (tensor<1x59x1x28xi1>) -> tensor<1x59x1x28xi1>
    %21 = tosa.abs %16 : (tensor<1x59x1x28xi1>) -> tensor<1x59x1x28xi1>
    %22 = tosa.reduce_all %21 {axis = 2 : i32} : (tensor<1x59x1x28xi1>) -> tensor<1x59x1x28xi1>
    return %11, %13, %14, %17, %19, %20, %22 : tensor<1x1x1xi32>, tensor<59x1x1x1xf32>, tensor<2x1x7x9x10x9xi1>, tensor<59x1x24x1xf32>, tensor<826x1x1xi1>, tensor<1x59x1x28xi1>, tensor<1x59x1x28xi1>
  }
}
