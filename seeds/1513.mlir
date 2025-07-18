module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<100x59xi16>, %arg3: tensor<100x90x3x27x96xi32>, %arg4: tensor<100x90x3x1x96xi32>, %arg5: tensor<86x88xi1>, %arg6: tensor<78x26x23x93xf32>) -> (tensor<100x1xi16>, tensor<i1>, tensor<100x90x3x27x96xi1>, tensor<1x1xi1>, tensor<86xi1>, tensor<78x26x23x93xf32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.reduce_max %arg2 {axis = 1 : i32} : (tensor<100x59xi16>) -> tensor<100x1xi16>
    %2 = tosa.sub %1, %1 : (tensor<100x1xi16>, tensor<100x1xi16>) -> tensor<100x1xi16>
    %3 = tosa.bitwise_or %2, %1 : (tensor<100x1xi16>, tensor<100x1xi16>) -> tensor<100x1xi16>
    %4 = tosa.logical_and %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.clz %4 : (tensor<i1>) -> tensor<i1>
    %6 = tosa.logical_and %5, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.greater %arg3, %arg4 : (tensor<100x90x3x27x96xi32>, tensor<100x90x3x1x96xi32>) -> tensor<100x90x3x27x96xi1>
    %8 = tosa.reduce_all %arg5 {axis = 1 : i32} : (tensor<86x88xi1>) -> tensor<86x1xi1>
    %9 = tosa.reduce_max %8 {axis = 0 : i32} : (tensor<86x1xi1>) -> tensor<1x1xi1>
    %r_10 = tosa.const_shape {values = dense<[ 86 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %10 = tosa.reshape %8, %r_10 : (tensor<86x1xi1>, !tosa.shape<1>) -> tensor<86xi1>
    %11 = tosa.exp %arg6 : (tensor<78x26x23x93xf32>) -> tensor<78x26x23x93xf32>
    %12 = tosa.log %11 : (tensor<78x26x23x93xf32>) -> tensor<78x26x23x93xf32>
    return %3, %6, %7, %9, %10, %12 : tensor<100x1xi16>, tensor<i1>, tensor<100x90x3x27x96xi1>, tensor<1x1xi1>, tensor<86xi1>, tensor<78x26x23x93xf32>
  }
}
