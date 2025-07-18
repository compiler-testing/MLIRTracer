module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<5xf32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> (tensor<i1>, tensor<i1>, tensor<5xf32>, tensor<1xf32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.bitwise_and %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.reciprocal %arg2 : (tensor<5xf32>) -> tensor<5xf32>
    %3 = tosa.intdiv %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = tosa.equal %3, %3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = tosa.logical_not %4 : (tensor<i1>) -> tensor<i1>
    %6 = tosa.logical_not %5 : (tensor<i1>) -> tensor<i1>
    %7 = tosa.reciprocal %2 : (tensor<5xf32>) -> tensor<5xf32>
    %8 = tosa.maximum %7, %7 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    %9 = tosa.floor %2 : (tensor<5xf32>) -> tensor<5xf32>
    %t_10 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %10 = tosa.tile %9, %t_10 : (tensor<5xf32>, !tosa.shape<1>) -> tensor<10xf32>
    %11 = tosa.reduce_max %10 {axis = 0 : i32} : (tensor<10xf32>) -> tensor<1xf32>
    %12 = tosa.tanh %8 : (tensor<5xf32>) -> tensor<5xf32>
    %13 = tosa.ceil %12 : (tensor<5xf32>) -> tensor<5xf32>
    %14 = tosa.rsqrt %11 : (tensor<1xf32>) -> tensor<1xf32>
    return %1, %6, %13, %14 : tensor<i1>, tensor<i1>, tensor<5xf32>, tensor<1xf32>
  }
}
