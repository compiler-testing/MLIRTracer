module {
  func.func @main(%arg0: tensor<43x25xf32>, %arg1: tensor<43x25xf32>, %arg2: tensor<20x42x2x53x24xi64>, %arg3: tensor<1x42x1x53x24xi64>) -> (tensor<43x25xf32>, tensor<20x42x2x53x24xi1>, tensor<20x42x2x53x24xi64>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<43x25xf32>, tensor<43x25xf32>) -> tensor<43x25xf32>
    %1 = tosa.bitwise_and %arg2, %arg3 : (tensor<20x42x2x53x24xi64>, tensor<1x42x1x53x24xi64>) -> tensor<20x42x2x53x24xi64>
    %2 = tosa.sub %1, %1 : (tensor<20x42x2x53x24xi64>, tensor<20x42x2x53x24xi64>) -> tensor<20x42x2x53x24xi64>
    %3 = tosa.sigmoid %0 : (tensor<43x25xf32>) -> tensor<43x25xf32>
    %4 = tosa.bitwise_xor %2, %2 : (tensor<20x42x2x53x24xi64>, tensor<20x42x2x53x24xi64>) -> tensor<20x42x2x53x24xi64>
    %5 = tosa.greater_equal %4, %4 : (tensor<20x42x2x53x24xi64>, tensor<20x42x2x53x24xi64>) -> tensor<20x42x2x53x24xi1>
    %6 = tosa.logical_or %5, %5 : (tensor<20x42x2x53x24xi1>, tensor<20x42x2x53x24xi1>) -> tensor<20x42x2x53x24xi1>
    %7 = tosa.bitwise_xor %2, %2 : (tensor<20x42x2x53x24xi64>, tensor<20x42x2x53x24xi64>) -> tensor<20x42x2x53x24xi64>
    return %3, %6, %7 : tensor<43x25xf32>, tensor<20x42x2x53x24xi1>, tensor<20x42x2x53x24xi64>
  }
}
