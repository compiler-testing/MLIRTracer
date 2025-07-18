module {
  func.func @main(%arg0: tensor<58x1xf32>, %arg1: tensor<93x2x10xi1>, %arg2: tensor<93x2x10xi1>) -> (tensor<58x1xf32>, tensor<93x2x10xi1>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<58x1xf32>) -> tensor<58x1xf32>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<93x2x10xi1>, tensor<93x2x10xi1>) -> tensor<93x2x10xi1>
    %2 = tosa.pow %0, %0 : (tensor<58x1xf32>, tensor<58x1xf32>) -> tensor<58x1xf32>
    %3 = tosa.abs %1 : (tensor<93x2x10xi1>) -> tensor<93x2x10xi1>
    %4 = tosa.reverse %3 {axis = 0 : i32} : (tensor<93x2x10xi1>) -> tensor<93x2x10xi1>
    return %2, %4 : tensor<58x1xf32>, tensor<93x2x10xi1>
  }
}
