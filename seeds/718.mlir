module {
  func.func @main(%arg0: tensor<27x34x64x21x45x87xf32>, %arg1: tensor<1x11xi32>) -> (tensor<27x34x64x21x45x87xi1>, tensor<27x34x64x21x45x87xf32>, tensor<1xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<27x34x64x21x45x87xf32>) -> tensor<27x34x64x21x45x87xf32>
    %1 = tosa.argmax %arg1 {axis = 1 : i32} : (tensor<1x11xi32>) -> tensor<1xi32>
    %2 = tosa.rsqrt %0 : (tensor<27x34x64x21x45x87xf32>) -> tensor<27x34x64x21x45x87xf32>
    %3 = tosa.greater_equal %1, %1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %4 = tosa.greater %2, %2 : (tensor<27x34x64x21x45x87xf32>, tensor<27x34x64x21x45x87xf32>) -> tensor<27x34x64x21x45x87xi1>
    %5 = tosa.logical_not %4 : (tensor<27x34x64x21x45x87xi1>) -> tensor<27x34x64x21x45x87xi1>
    %6 = tosa.reciprocal %2 : (tensor<27x34x64x21x45x87xf32>) -> tensor<27x34x64x21x45x87xf32>
    %7 = tosa.reduce_sum %3 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %5, %6, %7 : tensor<27x34x64x21x45x87xi1>, tensor<27x34x64x21x45x87xf32>, tensor<1xi1>
  }
}
