module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<78x83x26x50x36x19xi8>, %arg2: tensor<1x1x1x50x1x1xi8>, %arg3: tensor<18x6x84x31xi64>, %arg4: tensor<94xf32>) -> (tensor<i1>, tensor<18x6x1x31xi64>, tensor<94xf32>, tensor<78x83x26x50x36x19xi1>, tensor<78x83x26x50x36x19xi1>) {
    %0 = tosa.clz %arg0 : (tensor<i1>) -> tensor<i1>
    %1 = tosa.equal %arg1, %arg2 : (tensor<78x83x26x50x36x19xi8>, tensor<1x1x1x50x1x1xi8>) -> tensor<78x83x26x50x36x19xi1>
    %2 = tosa.reduce_max %arg3 {axis = 2 : i32} : (tensor<18x6x84x31xi64>) -> tensor<18x6x1x31xi64>
    %3 = tosa.reduce_product %2 {axis = 2 : i32} : (tensor<18x6x1x31xi64>) -> tensor<18x6x1x31xi64>
    %4 = tosa.add %3, %2 : (tensor<18x6x1x31xi64>, tensor<18x6x1x31xi64>) -> tensor<18x6x1x31xi64>
    %5 = tosa.sub %4, %3 : (tensor<18x6x1x31xi64>, tensor<18x6x1x31xi64>) -> tensor<18x6x1x31xi64>
    %6 = tosa.sigmoid %arg4 : (tensor<94xf32>) -> tensor<94xf32>
    %7 = tosa.logical_not %1 : (tensor<78x83x26x50x36x19xi1>) -> tensor<78x83x26x50x36x19xi1>
    %8 = tosa.logical_and %1, %1 : (tensor<78x83x26x50x36x19xi1>, tensor<78x83x26x50x36x19xi1>) -> tensor<78x83x26x50x36x19xi1>
    return %0, %5, %6, %7, %8 : tensor<i1>, tensor<18x6x1x31xi64>, tensor<94xf32>, tensor<78x83x26x50x36x19xi1>, tensor<78x83x26x50x36x19xi1>
  }
}
