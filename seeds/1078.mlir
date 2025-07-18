module {
  func.func @main(%arg0: tensor<19x5x14xi32>, %arg1: tensor<1x5x1xi32>, %arg2: tensor<12x88x11x86xf32>, %arg3: tensor<58x80x73xi1>) -> (tensor<12x88x11x86xf32>, tensor<19x5x14xi32>, tensor<1x80x73xi1>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<19x5x14xi32>, tensor<1x5x1xi32>) -> tensor<19x5x14xi32>
    %1 = tosa.ceil %arg2 : (tensor<12x88x11x86xf32>) -> tensor<12x88x11x86xf32>
    %2 = tosa.identity %0 : (tensor<19x5x14xi32>) -> tensor<19x5x14xi32>
    %3 = tosa.intdiv %2, %2 : (tensor<19x5x14xi32>, tensor<19x5x14xi32>) -> tensor<19x5x14xi32>
    %4 = tosa.sub %3, %3 : (tensor<19x5x14xi32>, tensor<19x5x14xi32>) -> tensor<19x5x14xi32>
    %5 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<58x80x73xi1>) -> tensor<1x80x73xi1>
    return %1, %4, %5 : tensor<12x88x11x86xf32>, tensor<19x5x14xi32>, tensor<1x80x73xi1>
  }
}
