module {
  func.func @main(%arg0: tensor<70x97x65x85xi1>, %arg1: tensor<69x20x10xf32>) -> (tensor<70x97x65x85xi1>, tensor<69x20x10xf32>) {
    %0 = tosa.abs %arg0 : (tensor<70x97x65x85xi1>) -> tensor<70x97x65x85xi1>
    %1 = tosa.identity %0 : (tensor<70x97x65x85xi1>) -> tensor<70x97x65x85xi1>
    %2 = tosa.reciprocal %arg1 : (tensor<69x20x10xf32>) -> tensor<69x20x10xf32>
    %3 = tosa.logical_left_shift %1, %0 : (tensor<70x97x65x85xi1>, tensor<70x97x65x85xi1>) -> tensor<70x97x65x85xi1>
    %4 = tosa.clamp %2 {min_val = 2.000000e+00 : f32, max_val = 7.100000e+01 : f32} : (tensor<69x20x10xf32>) -> tensor<69x20x10xf32>
    %5 = tosa.log %4 : (tensor<69x20x10xf32>) -> tensor<69x20x10xf32>
    return %3, %5 : tensor<70x97x65x85xi1>, tensor<69x20x10xf32>
  }
}
