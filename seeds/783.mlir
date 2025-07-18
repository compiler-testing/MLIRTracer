module {
  func.func @main(%arg0: tensor<27x37x63x30x84x31xi64>, %arg1: tensor<1x37x63x30x1x31xi64>, %arg2: tensor<50x6xf32>) -> (tensor<50x6xf32>, tensor<27x37x63x30x84x31xi1>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<27x37x63x30x84x31xi64>, tensor<1x37x63x30x1x31xi64>) -> tensor<27x37x63x30x84x31xi64>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<27x37x63x30x84x31xi64>, tensor<27x37x63x30x84x31xi64>) -> tensor<27x37x63x30x84x31xi64>
    %2 = tosa.equal %1, %0 : (tensor<27x37x63x30x84x31xi64>, tensor<27x37x63x30x84x31xi64>) -> tensor<27x37x63x30x84x31xi1>
    %3 = tosa.logical_not %2 : (tensor<27x37x63x30x84x31xi1>) -> tensor<27x37x63x30x84x31xi1>
    %4 = tosa.logical_right_shift %3, %3 : (tensor<27x37x63x30x84x31xi1>, tensor<27x37x63x30x84x31xi1>) -> tensor<27x37x63x30x84x31xi1>
    %5 = tosa.logical_xor %4, %2 : (tensor<27x37x63x30x84x31xi1>, tensor<27x37x63x30x84x31xi1>) -> tensor<27x37x63x30x84x31xi1>
    %6 = tosa.tanh %arg2 : (tensor<50x6xf32>) -> tensor<50x6xf32>
    %7 = tosa.bitwise_or %5, %3 : (tensor<27x37x63x30x84x31xi1>, tensor<27x37x63x30x84x31xi1>) -> tensor<27x37x63x30x84x31xi1>
    return %6, %7 : tensor<50x6xf32>, tensor<27x37x63x30x84x31xi1>
  }
}
