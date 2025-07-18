module {
  func.func @main(%arg0: tensor<20x68x67x24x48x78xf32>) -> tensor<20x68x67x24x48x78xf32> {
    %0 = tosa.reciprocal %arg0 : (tensor<20x68x67x24x48x78xf32>) -> tensor<20x68x67x24x48x78xf32>
    %1 = tosa.sigmoid %0 : (tensor<20x68x67x24x48x78xf32>) -> tensor<20x68x67x24x48x78xf32>
    return %1 : tensor<20x68x67x24x48x78xf32>
  }
}
