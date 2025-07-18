module {
  func.func @main(%arg0: tensor<49x51xi32>, %arg1: tensor<38x51xi32>, %arg2: tensor<4x22x79x62x87xi1>, %arg3: tensor<1x22x1x1x87xi1>) -> (tensor<4x22x79x62x87xi1>, tensor<87x51xi1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<49x51xi32>, tensor<38x51xi32>) -> tensor<87x51xi32>
    %1 = tosa.logical_xor %arg2, %arg3 : (tensor<4x22x79x62x87xi1>, tensor<1x22x1x1x87xi1>) -> tensor<4x22x79x62x87xi1>
    %2 = tosa.greater_equal %0, %0 : (tensor<87x51xi32>, tensor<87x51xi32>) -> tensor<87x51xi1>
    %3 = tosa.reverse %2 {axis = 1 : i32} : (tensor<87x51xi1>) -> tensor<87x51xi1>
    return %1, %3 : tensor<4x22x79x62x87xi1>, tensor<87x51xi1>
  }
}
