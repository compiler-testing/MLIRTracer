module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<26x75x66x20xi1>, %arg3: tensor<26x1x1x20xi1>) -> (tensor<f32>, tensor<26x75x66x20xi1>, tensor<f32>, tensor<26x75x66x20xi1>, tensor<f32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = tosa.logical_and %arg2, %arg3 : (tensor<26x75x66x20xi1>, tensor<26x1x1x20xi1>) -> tensor<26x75x66x20xi1>
    %2 = tosa.bitwise_not %1 : (tensor<26x75x66x20xi1>) -> tensor<26x75x66x20xi1>
    %3 = tosa.ceil %0 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.logical_or %1, %2 : (tensor<26x75x66x20xi1>, tensor<26x75x66x20xi1>) -> tensor<26x75x66x20xi1>
    %5 = tosa.exp %0 : (tensor<f32>) -> tensor<f32>
    %6 = tosa.add %4, %2 : (tensor<26x75x66x20xi1>, tensor<26x75x66x20xi1>) -> tensor<26x75x66x20xi1>
    %7 = tosa.logical_not %2 : (tensor<26x75x66x20xi1>) -> tensor<26x75x66x20xi1>
    %8 = tosa.logical_and %7, %2 : (tensor<26x75x66x20xi1>, tensor<26x75x66x20xi1>) -> tensor<26x75x66x20xi1>
    %9 = tosa.floor %5 : (tensor<f32>) -> tensor<f32>
    %10 = tosa.clz %6 : (tensor<26x75x66x20xi1>) -> tensor<26x75x66x20xi1>
    %11 = tosa.ceil %5 : (tensor<f32>) -> tensor<f32>
    return %3, %8, %9, %10, %11 : tensor<f32>, tensor<26x75x66x20xi1>, tensor<f32>, tensor<26x75x66x20xi1>, tensor<f32>
  }
}
