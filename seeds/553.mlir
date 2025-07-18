module {
  func.func @main(%arg0: tensor<17x70x16x47x60xi1>, %arg1: tensor<17x70x16x47x1xi1>, %arg2: tensor<54x19x22xf32>, %arg3: tensor<66x10x95x49xi1>) -> (tensor<17x70x16x47x60xi1>, tensor<1x1x22xf32>, tensor<66x10x1x49xi1>, tensor<54x19x22xi1>, tensor<1x66x49x10xi1>, tensor<66x20x1x49xi1>, tensor<1x19x22xf32>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<17x70x16x47x60xi1>, tensor<17x70x16x47x1xi1>) -> tensor<17x70x16x47x60xi1>
    %1 = tosa.reverse %arg2 {axis = 0 : i32} : (tensor<54x19x22xf32>) -> tensor<54x19x22xf32>
    %2 = tosa.reduce_any %arg3 {axis = 2 : i32} : (tensor<66x10x95x49xi1>) -> tensor<66x10x1x49xi1>
    %3 = tosa.bitwise_and %0, %0 : (tensor<17x70x16x47x60xi1>, tensor<17x70x16x47x60xi1>) -> tensor<17x70x16x47x60xi1>
    %4 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<54x19x22xf32>) -> tensor<1x19x22xf32>
    %5 = tosa.reduce_product %4 {axis = 1 : i32} : (tensor<1x19x22xf32>) -> tensor<1x1x22xf32>
    %6 = tosa.arithmetic_right_shift %2, %2 {round = true} : (tensor<66x10x1x49xi1>, tensor<66x10x1x49xi1>) -> tensor<66x10x1x49xi1>
    %7 = tosa.greater %1, %1 : (tensor<54x19x22xf32>, tensor<54x19x22xf32>) -> tensor<54x19x22xi1>
    %8 = "tosa.const"() {values = dense<[2, 0, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
    %9 = tosa.transpose %2 {perms = array<i32: 2, 0, 3, 1>} : (tensor<66x10x1x49xi1>) -> tensor<1x66x49x10xi1>
    %10 = tosa.concat %2, %2 {axis = 1 : i32} : (tensor<66x10x1x49xi1>, tensor<66x10x1x49xi1>) -> tensor<66x20x1x49xi1>
    %11 = tosa.pow %4, %4 : (tensor<1x19x22xf32>, tensor<1x19x22xf32>) -> tensor<1x19x22xf32>
    return %3, %5, %6, %7, %9, %10, %11 : tensor<17x70x16x47x60xi1>, tensor<1x1x22xf32>, tensor<66x10x1x49xi1>, tensor<54x19x22xi1>, tensor<1x66x49x10xi1>, tensor<66x20x1x49xi1>, tensor<1x19x22xf32>
  }
}
