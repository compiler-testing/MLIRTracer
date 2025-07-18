module {
  func.func @main(%arg0: tensor<69x13x89x29x47x90xi16>, %arg1: tensor<69x1x89x1x1x90xi16>, %arg2: tensor<43x55xf32>, %arg3: tensor<43x55xf32>, %arg4: tensor<9x49x42xi1>) -> (tensor<43x55xf32>, tensor<69x13x89x29x47x90xi16>, tensor<1x49x42xi1>, tensor<43xi32>, tensor<1xi1>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<69x13x89x29x47x90xi16>, tensor<69x1x89x1x1x90xi16>) -> tensor<69x13x89x29x47x90xi16>
    %1 = tosa.pow %arg2, %arg3 : (tensor<43x55xf32>, tensor<43x55xf32>) -> tensor<43x55xf32>
    %2 = tosa.rsqrt %1 : (tensor<43x55xf32>) -> tensor<43x55xf32>
    %3 = tosa.bitwise_xor %0, %0 : (tensor<69x13x89x29x47x90xi16>, tensor<69x13x89x29x47x90xi16>) -> tensor<69x13x89x29x47x90xi16>
    %4 = tosa.argmax %1 {axis = 1 : i32} : (tensor<43x55xf32>) -> tensor<43xi32>
    %5 = tosa.reduce_any %arg4 {axis = 0 : i32} : (tensor<9x49x42xi1>) -> tensor<1x49x42xi1>
    %6 = tosa.intdiv %4, %4 : (tensor<43xi32>, tensor<43xi32>) -> tensor<43xi32>
    %7 = tosa.greater %4, %4 : (tensor<43xi32>, tensor<43xi32>) -> tensor<43xi1>
    %8 = tosa.reduce_product %7 {axis = 0 : i32} : (tensor<43xi1>) -> tensor<1xi1>
    return %2, %3, %5, %6, %8 : tensor<43x55xf32>, tensor<69x13x89x29x47x90xi16>, tensor<1x49x42xi1>, tensor<43xi32>, tensor<1xi1>
  }
}
