module {
  func.func @main(%arg0: tensor<87x81x73x35x95xi32>, %arg1: tensor<87x1x73x35x95xi32>, %arg2: tensor<33x20x50x90xi32>, %arg3: tensor<33x1x50x1xi32>, %arg4: tensor<21x69x98xf32>) -> (tensor<87x81x73x35x95xi1>, tensor<1x20x50x90xi1>, tensor<21x69x98xf32>, tensor<21x69x98xf32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<87x81x73x35x95xi32>, tensor<87x1x73x35x95xi32>) -> tensor<87x81x73x35x95xi1>
    %1 = tosa.greater %arg2, %arg3 : (tensor<33x20x50x90xi32>, tensor<33x1x50x1xi32>) -> tensor<33x20x50x90xi1>
    %2 = tosa.logical_or %1, %1 : (tensor<33x20x50x90xi1>, tensor<33x20x50x90xi1>) -> tensor<33x20x50x90xi1>
    %3 = tosa.bitwise_and %0, %0 : (tensor<87x81x73x35x95xi1>, tensor<87x81x73x35x95xi1>) -> tensor<87x81x73x35x95xi1>
    %4 = tosa.logical_left_shift %2, %2 : (tensor<33x20x50x90xi1>, tensor<33x20x50x90xi1>) -> tensor<33x20x50x90xi1>
    %5 = tosa.logical_xor %4, %1 : (tensor<33x20x50x90xi1>, tensor<33x20x50x90xi1>) -> tensor<33x20x50x90xi1>
    %6 = tosa.exp %arg4 : (tensor<21x69x98xf32>) -> tensor<21x69x98xf32>
    %7 = tosa.reduce_any %5 {axis = 0 : i32} : (tensor<33x20x50x90xi1>) -> tensor<1x20x50x90xi1>
    %8 = tosa.reverse %6 {axis = 0 : i32} : (tensor<21x69x98xf32>) -> tensor<21x69x98xf32>
    %9 = tosa.pow %6, %6 : (tensor<21x69x98xf32>, tensor<21x69x98xf32>) -> tensor<21x69x98xf32>
    return %3, %7, %8, %9 : tensor<87x81x73x35x95xi1>, tensor<1x20x50x90xi1>, tensor<21x69x98xf32>, tensor<21x69x98xf32>
  }
}
