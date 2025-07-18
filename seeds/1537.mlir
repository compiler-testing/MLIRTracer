module {
  func.func @main(%arg0: tensor<33x14x83x5xf32>, %arg1: tensor<80x58xi1>) -> (tensor<1x58xi1>, tensor<1x14x1x10xf32>) {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<33x14x83x5xf32>) -> tensor<1x14x83x5xf32>
    %1 = tosa.concat %0, %0 {axis = 3 : i32} : (tensor<1x14x83x5xf32>, tensor<1x14x83x5xf32>) -> tensor<1x14x83x10xf32>
    %2 = tosa.reduce_product %1 {axis = 2 : i32} : (tensor<1x14x83x10xf32>) -> tensor<1x14x1x10xf32>
    %3 = tosa.reduce_min %2 {axis = 2 : i32} : (tensor<1x14x1x10xf32>) -> tensor<1x14x1x10xf32>
    %4 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<80x58xi1>) -> tensor<1x58xi1>
    %5 = tosa.tanh %3 : (tensor<1x14x1x10xf32>) -> tensor<1x14x1x10xf32>
    %6 = tosa.exp %5 : (tensor<1x14x1x10xf32>) -> tensor<1x14x1x10xf32>
    return %4, %6 : tensor<1x58xi1>, tensor<1x14x1x10xf32>
  }
}
