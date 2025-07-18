module {
  func.func @main(%arg0: tensor<23x100x89x26xi8>, %arg1: tensor<23x100x1x1xi8>, %arg2: tensor<55x94x87x46x88xf32>) -> (tensor<100x89x26xi32>, tensor<55x94x87x46x88xf32>, tensor<23x100x89x26xi1>, tensor<23x1x1x26xi1>, tensor<23x100x1x1xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<23x100x89x26xi8>, tensor<23x100x1x1xi8>) -> tensor<23x100x89x26xi1>
    %1 = tosa.sub %0, %0 : (tensor<23x100x89x26xi1>, tensor<23x100x89x26xi1>) -> tensor<23x100x89x26xi1>
    %2 = tosa.exp %arg2 : (tensor<55x94x87x46x88xf32>) -> tensor<55x94x87x46x88xf32>
    %3 = tosa.argmax %1 {axis = 0 : i32} : (tensor<23x100x89x26xi1>) -> tensor<100x89x26xi32>
    %4 = tosa.log %2 : (tensor<55x94x87x46x88xf32>) -> tensor<55x94x87x46x88xf32>
    %5 = tosa.add %2, %4 : (tensor<55x94x87x46x88xf32>, tensor<55x94x87x46x88xf32>) -> tensor<55x94x87x46x88xf32>
    %6 = tosa.reverse %1 {axis = 2 : i32} : (tensor<23x100x89x26xi1>) -> tensor<23x100x89x26xi1>
    %7 = tosa.reduce_max %1 {axis = 2 : i32} : (tensor<23x100x89x26xi1>) -> tensor<23x100x1x26xi1>
    %8 = tosa.logical_and %7, %7 : (tensor<23x100x1x26xi1>, tensor<23x100x1x26xi1>) -> tensor<23x100x1x26xi1>
    %9 = tosa.reduce_product %8 {axis = 1 : i32} : (tensor<23x100x1x26xi1>) -> tensor<23x1x1x26xi1>
    %10 = tosa.logical_not %8 : (tensor<23x100x1x26xi1>) -> tensor<23x100x1x26xi1>
    %11 = tosa.reduce_sum %10 {axis = 3 : i32} : (tensor<23x100x1x26xi1>) -> tensor<23x100x1x1xi1>
    return %3, %5, %6, %9, %11 : tensor<100x89x26xi32>, tensor<55x94x87x46x88xf32>, tensor<23x100x89x26xi1>, tensor<23x1x1x26xi1>, tensor<23x100x1x1xi1>
  }
}
