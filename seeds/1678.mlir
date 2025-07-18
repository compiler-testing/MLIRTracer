module {
  func.func @main(%arg0: tensor<85x44x24x66xf32>) -> tensor<1x44x24x66xf32> {
    %0 = tosa.exp %arg0 : (tensor<85x44x24x66xf32>) -> tensor<85x44x24x66xf32>
    %1 = tosa.abs %0 : (tensor<85x44x24x66xf32>) -> tensor<85x44x24x66xf32>
    %2 = tosa.sub %1, %0 : (tensor<85x44x24x66xf32>, tensor<85x44x24x66xf32>) -> tensor<85x44x24x66xf32>
    %3 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<85x44x24x66xf32>) -> tensor<1x44x24x66xf32>
    return %3 : tensor<1x44x24x66xf32>
  }
}
