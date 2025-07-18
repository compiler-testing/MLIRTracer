module {
  func.func @main(%arg0: tensor<42x99x78x27x24x75xf32>) -> tensor<42x99x78x27x24x75xf32> {
    %0 = tosa.ceil %arg0 : (tensor<42x99x78x27x24x75xf32>) -> tensor<42x99x78x27x24x75xf32>
    %1 = tosa.maximum %0, %0 : (tensor<42x99x78x27x24x75xf32>, tensor<42x99x78x27x24x75xf32>) -> tensor<42x99x78x27x24x75xf32>
    %2 = tosa.exp %1 : (tensor<42x99x78x27x24x75xf32>) -> tensor<42x99x78x27x24x75xf32>
    %3 = tosa.identity %2 : (tensor<42x99x78x27x24x75xf32>) -> tensor<42x99x78x27x24x75xf32>
    return %3 : tensor<42x99x78x27x24x75xf32>
  }
}
