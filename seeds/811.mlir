module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<49x31x89x22x57xf32>) -> (tensor<i32>, tensor<49x31x89x22x57xf32>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<i32>) -> tensor<i32>
    %1 = tosa.floor %arg1 : (tensor<49x31x89x22x57xf32>) -> tensor<49x31x89x22x57xf32>
    return %0, %1 : tensor<i32>, tensor<49x31x89x22x57xf32>
  }
}
