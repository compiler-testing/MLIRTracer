module {
  func.func @main(%arg0: tensor<66x10x17x72x17x89xf32>, %arg1: tensor<89x9xi32>, %arg2: tensor<61x75x16x57xi1>, %arg3: tensor<1x1x16x1xi1>) -> (tensor<61x75x16x57xi1>, tensor<89x9xi32>, tensor<89x9xi32>, tensor<1x9xi32>, tensor<66x10x17x72x17x89xf32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<66x10x17x72x17x89xf32>) -> tensor<66x10x17x72x17x89xf32>
    %1 = tosa.reverse %arg1 {axis = 1 : i32} : (tensor<89x9xi32>) -> tensor<89x9xi32>
    %2 = tosa.logical_or %arg2, %arg3 : (tensor<61x75x16x57xi1>, tensor<1x1x16x1xi1>) -> tensor<61x75x16x57xi1>
    %3 = tosa.intdiv %1, %1 : (tensor<89x9xi32>, tensor<89x9xi32>) -> tensor<89x9xi32>
    %4 = tosa.bitwise_not %3 : (tensor<89x9xi32>) -> tensor<89x9xi32>
    %5 = tosa.reduce_max %4 {axis = 0 : i32} : (tensor<89x9xi32>) -> tensor<1x9xi32>
    %6 = tosa.clamp %1 {min_val = -39 : i32, max_val = 22 : i32} : (tensor<89x9xi32>) -> tensor<89x9xi32>
    %7 = tosa.bitwise_or %1, %1 : (tensor<89x9xi32>, tensor<89x9xi32>) -> tensor<89x9xi32>
    %8 = tosa.bitwise_not %5 : (tensor<1x9xi32>) -> tensor<1x9xi32>
    %9 = tosa.reciprocal %0 : (tensor<66x10x17x72x17x89xf32>) -> tensor<66x10x17x72x17x89xf32>
    return %2, %6, %7, %8, %9 : tensor<61x75x16x57xi1>, tensor<89x9xi32>, tensor<89x9xi32>, tensor<1x9xi32>, tensor<66x10x17x72x17x89xf32>
  }
}
