module {
  func.func @main(%arg0: tensor<65x94x91x69x34xf32>, %arg1: tensor<1x1x91x69x1xf32>, %arg2: tensor<53x87x7x50xi1>, %arg3: tensor<53x87x1x1xi1>, %arg4: tensor<i32>, %arg5: tensor<i32>) -> (tensor<65x94x91x69x34xf32>, tensor<53x1x7x50xi1>, tensor<i32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<65x94x91x69x34xf32>, tensor<1x1x91x69x1xf32>) -> tensor<65x94x91x69x34xf32>
    %1 = tosa.ceil %0 : (tensor<65x94x91x69x34xf32>) -> tensor<65x94x91x69x34xf32>
    %2 = tosa.logical_or %arg2, %arg3 : (tensor<53x87x7x50xi1>, tensor<53x87x1x1xi1>) -> tensor<53x87x7x50xi1>
    %3 = tosa.reduce_sum %2 {axis = 1 : i32} : (tensor<53x87x7x50xi1>) -> tensor<53x1x7x50xi1>
    %4 = tosa.intdiv %arg4, %arg5 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %1, %3, %4 : tensor<65x94x91x69x34xf32>, tensor<53x1x7x50xi1>, tensor<i32>
  }
}
