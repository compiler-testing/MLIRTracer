module {
  func.func @main(%arg0: tensor<68xf32>, %arg1: tensor<1xf32>, %arg2: tensor<18x87x68xi32>, %arg3: tensor<18x87x1xi32>) -> (tensor<272xf32>, tensor<18x87x68xi32>, tensor<68xf32>, tensor<136xf32>, tensor<272xf32>, tensor<272xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<68xf32>, tensor<1xf32>) -> tensor<68xf32>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<68xf32>, tensor<68xf32>) -> tensor<136xf32>
    %t_2 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.tile %1, %t_2 : (tensor<136xf32>, !tosa.shape<1>) -> tensor<272xf32>
    %3 = tosa.greater_equal %2, %2 : (tensor<272xf32>, tensor<272xf32>) -> tensor<272xi1>
    %4 = tosa.reciprocal %2 : (tensor<272xf32>) -> tensor<272xf32>
    %5 = tosa.intdiv %arg2, %arg3 : (tensor<18x87x68xi32>, tensor<18x87x1xi32>) -> tensor<18x87x68xi32>
    %6 = tosa.log %0 : (tensor<68xf32>) -> tensor<68xf32>
    %7 = tosa.logical_not %3 : (tensor<272xi1>) -> tensor<272xi1>
    %8 = tosa.reciprocal %1 : (tensor<136xf32>) -> tensor<136xf32>
    %9 = tosa.floor %2 : (tensor<272xf32>) -> tensor<272xf32>
    %10 = tosa.logical_not %7 : (tensor<272xi1>) -> tensor<272xi1>
    return %4, %5, %6, %8, %9, %10 : tensor<272xf32>, tensor<18x87x68xi32>, tensor<68xf32>, tensor<136xf32>, tensor<272xf32>, tensor<272xi1>
  }
}
