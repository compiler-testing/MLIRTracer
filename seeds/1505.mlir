module {
  func.func @main(%arg0: tensor<54x46x45x62xi32>, %arg1: tensor<54x46x45x1xi32>, %arg2: tensor<83x91x47xi1>) -> (tensor<690x1116x9x1xi32>, tensor<83x91x47xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<54x46x45x62xi32>, tensor<54x46x45x1xi32>) -> tensor<54x46x45x62xi32>
    %1 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0, 3, 1, 2>} : (tensor<54x46x45x62xi32>) -> tensor<54x62x46x45xi32>
    %r_3 = tosa.const_shape {values = dense<[ 690, 1116, 9, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.reshape %2, %r_3 : (tensor<54x62x46x45xi32>, !tosa.shape<4>) -> tensor<690x1116x9x1xi32>
    %4 = tosa.logical_right_shift %3, %3 : (tensor<690x1116x9x1xi32>, tensor<690x1116x9x1xi32>) -> tensor<690x1116x9x1xi32>
    %5 = tosa.bitwise_xor %4, %3 : (tensor<690x1116x9x1xi32>, tensor<690x1116x9x1xi32>) -> tensor<690x1116x9x1xi32>
    %6 = tosa.reduce_max %5 {axis = 3 : i32} : (tensor<690x1116x9x1xi32>) -> tensor<690x1116x9x1xi32>
    %7 = tosa.logical_not %arg2 : (tensor<83x91x47xi1>) -> tensor<83x91x47xi1>
    %8 = tosa.logical_not %7 : (tensor<83x91x47xi1>) -> tensor<83x91x47xi1>
    return %6, %8 : tensor<690x1116x9x1xi32>, tensor<83x91x47xi1>
  }
}
