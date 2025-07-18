module {
  func.func @main(%arg0: tensor<55x21x59x85x58x16xi16>, %arg1: tensor<52x95x77x88x24x17xi32>, %arg2: tensor<52x1x1x1x24x1xi32>, %arg3: tensor<13xi1>, %arg4: tensor<91x42xf32>) -> (tensor<52x95x77x88x24x17xi32>, tensor<55x21x59x85x58x16xi16>, tensor<91x42xf32>, tensor<1xi1>, tensor<1xi1>, tensor<1xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<55x21x59x85x58x16xi16>) -> tensor<55x21x59x85x58x16xi16>
    %1 = tosa.minimum %arg1, %arg2 : (tensor<52x95x77x88x24x17xi32>, tensor<52x1x1x1x24x1xi32>) -> tensor<52x95x77x88x24x17xi32>
    %2 = tosa.logical_left_shift %0, %0 : (tensor<55x21x59x85x58x16xi16>, tensor<55x21x59x85x58x16xi16>) -> tensor<55x21x59x85x58x16xi16>
    %3 = tosa.reduce_max %arg3 {axis = 0 : i32} : (tensor<13xi1>) -> tensor<1xi1>
    %4 = tosa.reduce_all %3 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.sigmoid %arg4 : (tensor<91x42xf32>) -> tensor<91x42xf32>
    %6 = tosa.reduce_any %4 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.logical_xor %3, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %8 = tosa.reverse %3 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %1, %2, %5, %6, %7, %8 : tensor<52x95x77x88x24x17xi32>, tensor<55x21x59x85x58x16xi16>, tensor<91x42xf32>, tensor<1xi1>, tensor<1xi1>, tensor<1xi1>
  }
}
