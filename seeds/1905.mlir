module {
  func.func @main(%arg0: tensor<17x31x94xi32>, %arg1: tensor<85xf32>) -> (tensor<17x1x94xi1>, tensor<85xf32>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<17x31x94xi32>) -> tensor<17x31x94xi32>
    %1 = tosa.greater_equal %0, %0 : (tensor<17x31x94xi32>, tensor<17x31x94xi32>) -> tensor<17x31x94xi1>
    %2 = tosa.reduce_all %1 {axis = 1 : i32} : (tensor<17x31x94xi1>) -> tensor<17x1x94xi1>
    %3 = tosa.logical_and %2, %2 : (tensor<17x1x94xi1>, tensor<17x1x94xi1>) -> tensor<17x1x94xi1>
    %4 = tosa.reverse %3 {axis = 2 : i32} : (tensor<17x1x94xi1>) -> tensor<17x1x94xi1>
    %5 = tosa.clz %4 : (tensor<17x1x94xi1>) -> tensor<17x1x94xi1>
    %6 = tosa.exp %arg1 : (tensor<85xf32>) -> tensor<85xf32>
    return %5, %6 : tensor<17x1x94xi1>, tensor<85xf32>
  }
}
