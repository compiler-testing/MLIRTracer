module {
  func.func @main(%arg0: tensor<17x12xi1>, %arg1: tensor<88xf32>, %arg2: tensor<1xf32>) -> (tensor<1x12xi1>, tensor<88xf32>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<17x12xi1>) -> tensor<1x12xi1>
    %1 = tosa.pow %arg1, %arg2 : (tensor<88xf32>, tensor<1xf32>) -> tensor<88xf32>
    return %0, %1 : tensor<1x12xi1>, tensor<88xf32>
  }
}
