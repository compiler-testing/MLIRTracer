module {
  func.func @main(%arg0: tensor<49x73x85xi32>, %arg1: tensor<1x73x1xi32>) -> tensor<49x1x85xi32> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<49x73x85xi32>, tensor<1x73x1xi32>) -> tensor<49x73x85xi32>
    %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<49x73x85xi32>) -> tensor<49x1x85xi32>
    %2 = tosa.bitwise_and %1, %1 : (tensor<49x1x85xi32>, tensor<49x1x85xi32>) -> tensor<49x1x85xi32>
    %3 = tosa.add %2, %2 : (tensor<49x1x85xi32>, tensor<49x1x85xi32>) -> tensor<49x1x85xi32>
    return %3 : tensor<49x1x85xi32>
  }
}
