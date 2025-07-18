module {
  func.func @main(%arg0: tensor<69x9x40x14x38xf32>, %arg1: tensor<65x50x62x66xf32>, %arg2: tensor<55x53x19xi32>, %arg3: tensor<1x1x1xi32>) -> (tensor<65x50x62x1xf32>, tensor<69x9x40x14x38xf32>, tensor<55x53x19xi1>) {
    %0 = tosa.exp %arg0 : (tensor<69x9x40x14x38xf32>) -> tensor<69x9x40x14x38xf32>
    %1 = tosa.maximum %0, %0 : (tensor<69x9x40x14x38xf32>, tensor<69x9x40x14x38xf32>) -> tensor<69x9x40x14x38xf32>
    %2 = tosa.minimum %1, %0 : (tensor<69x9x40x14x38xf32>, tensor<69x9x40x14x38xf32>) -> tensor<69x9x40x14x38xf32>
    %3 = tosa.maximum %2, %1 : (tensor<69x9x40x14x38xf32>, tensor<69x9x40x14x38xf32>) -> tensor<69x9x40x14x38xf32>
    %4 = tosa.minimum %3, %2 : (tensor<69x9x40x14x38xf32>, tensor<69x9x40x14x38xf32>) -> tensor<69x9x40x14x38xf32>
    %5 = tosa.reduce_sum %arg1 {axis = 3 : i32} : (tensor<65x50x62x66xf32>) -> tensor<65x50x62x1xf32>
    %6 = tosa.minimum %5, %5 : (tensor<65x50x62x1xf32>, tensor<65x50x62x1xf32>) -> tensor<65x50x62x1xf32>
    %7 = tosa.rsqrt %4 : (tensor<69x9x40x14x38xf32>) -> tensor<69x9x40x14x38xf32>
    %8 = tosa.intdiv %arg2, %arg3 : (tensor<55x53x19xi32>, tensor<1x1x1xi32>) -> tensor<55x53x19xi32>
    %9 = tosa.equal %8, %8 : (tensor<55x53x19xi32>, tensor<55x53x19xi32>) -> tensor<55x53x19xi1>
    return %6, %7, %9 : tensor<65x50x62x1xf32>, tensor<69x9x40x14x38xf32>, tensor<55x53x19xi1>
  }
}
