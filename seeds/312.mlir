module {
  func.func @main(%arg0: tensor<5xi64>, %arg1: tensor<5xi64>, %arg2: tensor<13x74x86xf32>) -> (tensor<5xi64>, tensor<13x74x86xf32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi64>
    %1 = tosa.sigmoid %arg2 : (tensor<13x74x86xf32>) -> tensor<13x74x86xf32>
    %2 = tosa.maximum %1, %1 : (tensor<13x74x86xf32>, tensor<13x74x86xf32>) -> tensor<13x74x86xf32>
    %3 = tosa.maximum %2, %1 : (tensor<13x74x86xf32>, tensor<13x74x86xf32>) -> tensor<13x74x86xf32>
    return %0, %3 : tensor<5xi64>, tensor<13x74x86xf32>
  }
}
