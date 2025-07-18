module {
  func.func @main(%arg0: tensor<45x83x70x46xi32>, %arg1: tensor<90x7x83x9x4x7xf32>) -> (tensor<45x1x70x46xi32>, tensor<90x7x83x9x4x7xf32>) {
    %0 = tosa.reduce_min %arg0 {axis = 1 : i32} : (tensor<45x83x70x46xi32>) -> tensor<45x1x70x46xi32>
    %1 = tosa.floor %arg1 : (tensor<90x7x83x9x4x7xf32>) -> tensor<90x7x83x9x4x7xf32>
    return %0, %1 : tensor<45x1x70x46xi32>, tensor<90x7x83x9x4x7xf32>
  }
}
