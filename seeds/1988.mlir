module {
  func.func @main(%arg0: tensor<7x27x28x50x65xf32>, %arg1: tensor<70x26x8xi1>) -> (tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>, tensor<13x16xi32>) {
    %0 = tosa.floor %arg0 : (tensor<7x27x28x50x65xf32>) -> tensor<7x27x28x50x65xf32>
    %1 = tosa.add %0, %0 : (tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>) -> tensor<7x27x28x50x65xf32>
    %2 = tosa.tanh %1 : (tensor<7x27x28x50x65xf32>) -> tensor<7x27x28x50x65xf32>
    %3 = tosa.logical_not %arg1 : (tensor<70x26x8xi1>) -> tensor<70x26x8xi1>
    %4 = tosa.bitwise_and %3, %3 : (tensor<70x26x8xi1>, tensor<70x26x8xi1>) -> tensor<70x26x8xi1>
    %5 = tosa.exp %2 : (tensor<7x27x28x50x65xf32>) -> tensor<7x27x28x50x65xf32>
    %6 = tosa.reduce_min %4 {axis = 0 : i32} : (tensor<70x26x8xi1>) -> tensor<1x26x8xi1>
    %r_7 = tosa.const_shape {values = dense<[ 1, 13, 16 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %7 = tosa.reshape %6, %r_7 : (tensor<1x26x8xi1>, !tosa.shape<3>) -> tensor<1x13x16xi1>
    %8 = tosa.exp %2 : (tensor<7x27x28x50x65xf32>) -> tensor<7x27x28x50x65xf32>
    %9 = tosa.minimum %5, %8 : (tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>) -> tensor<7x27x28x50x65xf32>
    %10 = tosa.minimum %5, %8 : (tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>) -> tensor<7x27x28x50x65xf32>
    %11 = tosa.exp %8 : (tensor<7x27x28x50x65xf32>) -> tensor<7x27x28x50x65xf32>
    %12 = tosa.maximum %0, %11 : (tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>) -> tensor<7x27x28x50x65xf32>
    %13 = tosa.minimum %8, %11 : (tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>) -> tensor<7x27x28x50x65xf32>
    %14 = tosa.reciprocal %8 : (tensor<7x27x28x50x65xf32>) -> tensor<7x27x28x50x65xf32>
    %15 = tosa.argmax %7 {axis = 0 : i32} : (tensor<1x13x16xi1>) -> tensor<13x16xi32>
    return %9, %10, %12, %13, %14, %15 : tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>, tensor<7x27x28x50x65xf32>, tensor<13x16xi32>
  }
}
