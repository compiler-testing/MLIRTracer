module {
  func.func @main(%arg0: tensor<2x72x82x98x76x34xi1>, %arg1: tensor<16x15xi32>, %arg2: tensor<21xf32>, %arg3: tensor<84x44x63xi1>) -> (tensor<2x72x82x98x76x34xi1>, tensor<48x30xi32>, tensor<5x7x9xi1>, tensor<i32>, tensor<21xf32>, tensor<21xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<2x72x82x98x76x34xi1>) -> tensor<2x72x82x98x76x34xi1>
    %t_1 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %arg1, %t_1 : (tensor<16x15xi32>, !tosa.shape<2>) -> tensor<48x30xi32>
    %2 = tosa.bitwise_and %1, %1 : (tensor<48x30xi32>, tensor<48x30xi32>) -> tensor<48x30xi32>
    %3 = tosa.reciprocal %arg2 : (tensor<21xf32>) -> tensor<21xf32>
    %4 = tosa.reduce_all %arg3 {axis = 2 : i32} : (tensor<84x44x63xi1>) -> tensor<84x44x1xi1>
    %5 = tosa.intdiv %2, %2 : (tensor<48x30xi32>, tensor<48x30xi32>) -> tensor<48x30xi32>
    %6 = tosa.sub %4, %4 : (tensor<84x44x1xi1>, tensor<84x44x1xi1>) -> tensor<84x44x1xi1>
    %7 = tosa.floor %3 : (tensor<21xf32>) -> tensor<21xf32>
    %8 = tosa.rsqrt %3 : (tensor<21xf32>) -> tensor<21xf32>
    %s_9_start = tosa.const_shape {values = dense<[ 79, 37, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_9_size = tosa.const_shape {values = dense<[ 5, 7, 9 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = tosa.slice %6, %s_9_start, %s_9_size : (tensor<84x44x1xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<5x7x9xi1>
    %10 = tosa.argmax %7 {axis = 0 : i32} : (tensor<21xf32>) -> tensor<i32>
    %11 = tosa.tanh %8 : (tensor<21xf32>) -> tensor<21xf32>
    %12 = tosa.floor %3 : (tensor<21xf32>) -> tensor<21xf32>
    return %0, %5, %9, %10, %11, %12 : tensor<2x72x82x98x76x34xi1>, tensor<48x30xi32>, tensor<5x7x9xi1>, tensor<i32>, tensor<21xf32>, tensor<21xf32>
  }
}
