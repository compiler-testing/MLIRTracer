module {
  func.func @main(%arg0: tensor<53x37x83x14x65x6xf32>, %arg1: tensor<53x37x1x1x65x1xf32>) -> tensor<53x74x83x14x65x6xf32> {
    %0 = tosa.sub %arg0, %arg1 : (tensor<53x37x83x14x65x6xf32>, tensor<53x37x1x1x65x1xf32>) -> tensor<53x37x83x14x65x6xf32>
    %1 = tosa.sub %0, %0 : (tensor<53x37x83x14x65x6xf32>, tensor<53x37x83x14x65x6xf32>) -> tensor<53x37x83x14x65x6xf32>
    %2 = tosa.exp %1 : (tensor<53x37x83x14x65x6xf32>) -> tensor<53x37x83x14x65x6xf32>
    %3 = tosa.exp %2 : (tensor<53x37x83x14x65x6xf32>) -> tensor<53x37x83x14x65x6xf32>
    %4 = tosa.pow %3, %2 : (tensor<53x37x83x14x65x6xf32>, tensor<53x37x83x14x65x6xf32>) -> tensor<53x37x83x14x65x6xf32>
    %5 = tosa.concat %4, %2 {axis = 1 : i32} : (tensor<53x37x83x14x65x6xf32>, tensor<53x37x83x14x65x6xf32>) -> tensor<53x74x83x14x65x6xf32>
    return %5 : tensor<53x74x83x14x65x6xf32>
  }
}
