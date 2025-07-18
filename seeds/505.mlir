module {
  func.func @main(%arg0: tensor<5x100x1x14x34xi64>, %arg1: tensor<5x100x1x14x34xi64>, %arg2: tensor<5x5x54x85x97xf32>, %arg3: tensor<63x54x29x64xi1>) -> (tensor<5x100x1x14x34xi1>, tensor<5x5x54x85x97xf32>, tensor<97x85x5x54x5xf32>, tensor<63x54x29x64xi1>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<5x100x1x14x34xi64>, tensor<5x100x1x14x34xi64>) -> tensor<5x100x1x14x34xi1>
    %1 = tosa.floor %arg2 : (tensor<5x5x54x85x97xf32>) -> tensor<5x5x54x85x97xf32>
    %2 = tosa.floor %1 : (tensor<5x5x54x85x97xf32>) -> tensor<5x5x54x85x97xf32>
    %3 = tosa.bitwise_or %0, %0 : (tensor<5x100x1x14x34xi1>, tensor<5x100x1x14x34xi1>) -> tensor<5x100x1x14x34xi1>
    %4 = tosa.maximum %2, %2 : (tensor<5x5x54x85x97xf32>, tensor<5x5x54x85x97xf32>) -> tensor<5x5x54x85x97xf32>
    %5 = tosa.exp %1 : (tensor<5x5x54x85x97xf32>) -> tensor<5x5x54x85x97xf32>
    %6 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %7 = tosa.transpose %4 {perms = array<i32: 4, 3, 0, 2, 1>} : (tensor<5x5x54x85x97xf32>) -> tensor<97x85x5x54x5xf32>
    %8 = tosa.reverse %arg3 {axis = 3 : i32} : (tensor<63x54x29x64xi1>) -> tensor<63x54x29x64xi1>
    %9 = tosa.logical_left_shift %8, %8 : (tensor<63x54x29x64xi1>, tensor<63x54x29x64xi1>) -> tensor<63x54x29x64xi1>
    return %3, %5, %7, %9 : tensor<5x100x1x14x34xi1>, tensor<5x5x54x85x97xf32>, tensor<97x85x5x54x5xf32>, tensor<63x54x29x64xi1>
  }
}
