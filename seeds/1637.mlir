module {
  func.func @main(%arg0: tensor<34x34x16x22x14x91xi1>, %arg1: tensor<1x34x1x1x14x1xi1>, %arg2: tensor<11xf32>, %arg3: tensor<36x42xi1>) -> (tensor<34x34x16x22x14x91xi1>, tensor<1xf32>, tensor<1x42xi1>, tensor<1x1xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<34x34x16x22x14x91xi1>, tensor<1x34x1x1x14x1xi1>) -> tensor<34x34x16x22x14x91xi1>
    %1 = tosa.bitwise_and %0, %0 : (tensor<34x34x16x22x14x91xi1>, tensor<34x34x16x22x14x91xi1>) -> tensor<34x34x16x22x14x91xi1>
    %2 = tosa.reduce_sum %arg2 {axis = 0 : i32} : (tensor<11xf32>) -> tensor<1xf32>
    %3 = tosa.reduce_any %arg3 {axis = 0 : i32} : (tensor<36x42xi1>) -> tensor<1x42xi1>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<1x42xi1>, tensor<1x42xi1>) -> tensor<1x42xi1>
    %5 = tosa.reduce_any %3 {axis = 1 : i32} : (tensor<1x42xi1>) -> tensor<1x1xi1>
    return %1, %2, %4, %5 : tensor<34x34x16x22x14x91xi1>, tensor<1xf32>, tensor<1x42xi1>, tensor<1x1xi1>
  }
}
