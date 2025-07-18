module {
  func.func @main(%arg0: tensor<75x89x53x77x80xi1>, %arg1: tensor<27x35x7x29x37xi32>, %arg2: tensor<1x35x7x29x1xi32>, %arg3: tensor<87xf32>) -> (tensor<75x89x53x77x80xi1>, tensor<87xf32>, tensor<27x35x7x29x74xi32>, tensor<27x35x7x29x37xi32>) {
    %0 = tosa.logical_not %arg0 : (tensor<75x89x53x77x80xi1>) -> tensor<75x89x53x77x80xi1>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<75x89x53x77x80xi1>, tensor<75x89x53x77x80xi1>) -> tensor<75x89x53x77x80xi1>
    %2 = tosa.intdiv %arg1, %arg2 : (tensor<27x35x7x29x37xi32>, tensor<1x35x7x29x1xi32>) -> tensor<27x35x7x29x37xi32>
    %3 = tosa.reciprocal %arg3 : (tensor<87xf32>) -> tensor<87xf32>
    %4 = tosa.concat %2, %2 {axis = 4 : i32} : (tensor<27x35x7x29x37xi32>, tensor<27x35x7x29x37xi32>) -> tensor<27x35x7x29x74xi32>
    %5 = tosa.logical_right_shift %2, %2 : (tensor<27x35x7x29x37xi32>, tensor<27x35x7x29x37xi32>) -> tensor<27x35x7x29x37xi32>
    return %1, %3, %4, %5 : tensor<75x89x53x77x80xi1>, tensor<87xf32>, tensor<27x35x7x29x74xi32>, tensor<27x35x7x29x37xi32>
  }
}
