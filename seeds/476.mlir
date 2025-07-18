module {
  func.func @main(%arg0: tensor<81x81x98x70x91xi64>, %arg1: tensor<1x1x98x1x91xi64>, %arg2: tensor<44x67x17x17x80x42xf32>, %arg3: tensor<90x86xi1>, %arg4: tensor<90x1xi1>) -> (tensor<81x81x98x70x91xi64>, tensor<90x86xi1>, tensor<44x67x17x17x80x42xf32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<81x81x98x70x91xi64>, tensor<1x1x98x1x91xi64>) -> tensor<81x81x98x70x91xi64>
    %1 = tosa.sigmoid %arg2 : (tensor<44x67x17x17x80x42xf32>) -> tensor<44x67x17x17x80x42xf32>
    %2 = tosa.logical_and %arg3, %arg4 : (tensor<90x86xi1>, tensor<90x1xi1>) -> tensor<90x86xi1>
    %3 = tosa.rsqrt %1 : (tensor<44x67x17x17x80x42xf32>) -> tensor<44x67x17x17x80x42xf32>
    %4 = tosa.minimum %3, %3 : (tensor<44x67x17x17x80x42xf32>, tensor<44x67x17x17x80x42xf32>) -> tensor<44x67x17x17x80x42xf32>
    return %0, %2, %4 : tensor<81x81x98x70x91xi64>, tensor<90x86xi1>, tensor<44x67x17x17x80x42xf32>
  }
}
