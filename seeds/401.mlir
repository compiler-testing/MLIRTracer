module {
  func.func @main(%arg0: tensor<29x71x83x11xi32>, %arg1: tensor<1x1x1x11xi32>, %arg2: tensor<55xi8>, %arg3: tensor<55xi8>, %arg4: tensor<75x57x18x14xi32>, %arg5: tensor<1x1x1x14xi32>, %arg6: tensor<24x10x9x21x100xf32>) -> (tensor<55xi1>, tensor<29x71x1x11xi1>, tensor<75x57x18x14xi1>, tensor<24x10x9x21x100xf32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<29x71x83x11xi32>, tensor<1x1x1x11xi32>) -> tensor<29x71x83x11xi1>
    %1 = tosa.greater %arg2, %arg3 : (tensor<55xi8>, tensor<55xi8>) -> tensor<55xi1>
    %2 = tosa.reduce_any %0 {axis = 2 : i32} : (tensor<29x71x83x11xi1>) -> tensor<29x71x1x11xi1>
    %3 = tosa.intdiv %arg4, %arg5 : (tensor<75x57x18x14xi32>, tensor<1x1x1x14xi32>) -> tensor<75x57x18x14xi32>
    %4 = tosa.greater %3, %3 : (tensor<75x57x18x14xi32>, tensor<75x57x18x14xi32>) -> tensor<75x57x18x14xi1>
    %5 = tosa.floor %arg6 : (tensor<24x10x9x21x100xf32>) -> tensor<24x10x9x21x100xf32>
    return %1, %2, %4, %5 : tensor<55xi1>, tensor<29x71x1x11xi1>, tensor<75x57x18x14xi1>, tensor<24x10x9x21x100xf32>
  }
}
