module {
  func.func @main(%arg0: tensor<98x49x75x17xf32>, %arg1: tensor<1x49x1x17xf32>, %arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<68xi32>, %arg5: tensor<68xi32>) -> (tensor<i1>, tensor<98x49x75x17xi1>, tensor<68xi32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<98x49x75x17xf32>, tensor<1x49x1x17xf32>) -> tensor<98x49x75x17xi1>
    %1 = tosa.greater %arg2, %arg3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %2 = tosa.intdiv %arg4, %arg5 : (tensor<68xi32>, tensor<68xi32>) -> tensor<68xi32>
    %r_3 = tosa.const_shape {values = dense<[ 68 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.reshape %2, %r_3 : (tensor<68xi32>, !tosa.shape<1>) -> tensor<68xi32>
    %4 = tosa.identity %0 : (tensor<98x49x75x17xi1>) -> tensor<98x49x75x17xi1>
    %5 = tosa.reverse %3 {axis = 0 : i32} : (tensor<68xi32>) -> tensor<68xi32>
    %6 = tosa.sub %5, %3 : (tensor<68xi32>, tensor<68xi32>) -> tensor<68xi32>
    return %1, %4, %6 : tensor<i1>, tensor<98x49x75x17xi1>, tensor<68xi32>
  }
}
