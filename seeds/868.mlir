module {
  func.func @main(%arg0: tensor<66x77x2x10x71xf32>, %arg1: tensor<86x52x74xi1>, %arg2: tensor<86x1x74xi1>) -> (tensor<66x77x2x10x71xf32>, tensor<86x74xi32>, tensor<86x52x74xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<66x77x2x10x71xf32>) -> tensor<66x77x2x10x71xf32>
    %1 = tosa.sub %0, %0 : (tensor<66x77x2x10x71xf32>, tensor<66x77x2x10x71xf32>) -> tensor<66x77x2x10x71xf32>
    %2 = tosa.add %1, %0 : (tensor<66x77x2x10x71xf32>, tensor<66x77x2x10x71xf32>) -> tensor<66x77x2x10x71xf32>
    %3 = tosa.identity %2 : (tensor<66x77x2x10x71xf32>) -> tensor<66x77x2x10x71xf32>
    %4 = tosa.sub %3, %3 : (tensor<66x77x2x10x71xf32>, tensor<66x77x2x10x71xf32>) -> tensor<66x77x2x10x71xf32>
    %5 = tosa.logical_xor %arg1, %arg2 : (tensor<86x52x74xi1>, tensor<86x1x74xi1>) -> tensor<86x52x74xi1>
    %6 = tosa.rsqrt %4 : (tensor<66x77x2x10x71xf32>) -> tensor<66x77x2x10x71xf32>
    %7 = tosa.argmax %5 {axis = 1 : i32} : (tensor<86x52x74xi1>) -> tensor<86x74xi32>
    %8 = tosa.logical_not %5 : (tensor<86x52x74xi1>) -> tensor<86x52x74xi1>
    return %6, %7, %8 : tensor<66x77x2x10x71xf32>, tensor<86x74xi32>, tensor<86x52x74xi1>
  }
}
