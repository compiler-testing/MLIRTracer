module {
  func.func @main(%arg0: tensor<35x14x4x30x22xf32>, %arg1: tensor<10x70xi8>) -> (tensor<1x70xi8>, tensor<35x14x4x30x22xf32>) {
    %0 = tosa.floor %arg0 : (tensor<35x14x4x30x22xf32>) -> tensor<35x14x4x30x22xf32>
    %1 = tosa.reduce_max %arg1 {axis = 0 : i32} : (tensor<10x70xi8>) -> tensor<1x70xi8>
    %2 = tosa.exp %0 : (tensor<35x14x4x30x22xf32>) -> tensor<35x14x4x30x22xf32>
    return %1, %2 : tensor<1x70xi8>, tensor<35x14x4x30x22xf32>
  }
}
