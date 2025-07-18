module {
  func.func @main(%arg0: tensor<29xi32>, %arg1: tensor<45xi1>, %arg2: tensor<1xi1>) -> (tensor<1xi32>, tensor<45xi1>) {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<29xi32>) -> tensor<1xi32>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<45xi1>, tensor<1xi1>) -> tensor<45xi1>
    return %0, %1 : tensor<1xi32>, tensor<45xi1>
  }
}
