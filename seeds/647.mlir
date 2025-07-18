module {
  func.func @main(%arg0: tensor<23x45x52x42xi32>, %arg1: tensor<23x1x52x1xi32>, %arg2: tensor<8x11x60xf32>, %arg3: tensor<1x11x60xf32>) -> (tensor<8x60xi32>, tensor<23x45x1x1xi1>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<23x45x52x42xi32>, tensor<23x1x52x1xi32>) -> tensor<23x45x52x42xi1>
    %1 = tosa.pow %arg2, %arg3 : (tensor<8x11x60xf32>, tensor<1x11x60xf32>) -> tensor<8x11x60xf32>
    %2 = tosa.add %0, %0 : (tensor<23x45x52x42xi1>, tensor<23x45x52x42xi1>) -> tensor<23x45x52x42xi1>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<23x45x52x42xi1>, tensor<23x45x52x42xi1>) -> tensor<23x45x52x42xi1>
    %4 = tosa.argmax %1 {axis = 1 : i32} : (tensor<8x11x60xf32>) -> tensor<8x60xi32>
    %5 = tosa.add %4, %4 : (tensor<8x60xi32>, tensor<8x60xi32>) -> tensor<8x60xi32>
    %6 = tosa.arithmetic_right_shift %3, %2 {round = true} : (tensor<23x45x52x42xi1>, tensor<23x45x52x42xi1>) -> tensor<23x45x52x42xi1>
    %7 = tosa.reduce_all %6 {axis = 3 : i32} : (tensor<23x45x52x42xi1>) -> tensor<23x45x52x1xi1>
    %8 = tosa.reduce_all %7 {axis = 2 : i32} : (tensor<23x45x52x1xi1>) -> tensor<23x45x1x1xi1>
    %9 = tosa.logical_not %8 : (tensor<23x45x1x1xi1>) -> tensor<23x45x1x1xi1>
    return %5, %9 : tensor<8x60xi32>, tensor<23x45x1x1xi1>
  }
}
