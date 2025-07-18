module {
  func.func @main(%arg0: tensor<48x95x9x84x26x25xi64>, %arg1: tensor<48x1x9x1x1x25xi64>, %arg2: tensor<93x66xf32>) -> (tensor<48x95x9x84x26x25xi1>, tensor<93x66xf32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<48x95x9x84x26x25xi64>, tensor<48x1x9x1x1x25xi64>) -> tensor<48x95x9x84x26x25xi1>
    %1 = tosa.sigmoid %arg2 : (tensor<93x66xf32>) -> tensor<93x66xf32>
    %2 = tosa.sigmoid %1 : (tensor<93x66xf32>) -> tensor<93x66xf32>
    %3 = tosa.rsqrt %2 : (tensor<93x66xf32>) -> tensor<93x66xf32>
    %4 = tosa.minimum %3, %3 : (tensor<93x66xf32>, tensor<93x66xf32>) -> tensor<93x66xf32>
    %5 = tosa.log %1 : (tensor<93x66xf32>) -> tensor<93x66xf32>
    %6 = tosa.sigmoid %5 : (tensor<93x66xf32>) -> tensor<93x66xf32>
    %7 = tosa.add %6, %4 : (tensor<93x66xf32>, tensor<93x66xf32>) -> tensor<93x66xf32>
    return %0, %7 : tensor<48x95x9x84x26x25xi1>, tensor<93x66xf32>
  }
}
