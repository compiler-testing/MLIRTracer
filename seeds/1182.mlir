module {
  func.func @main(%arg0: tensor<61xf32>, %arg1: tensor<1xf32>, %arg2: tensor<31x83x59x43x85x70xi8>, %arg3: tensor<31x1x1x43x1x70xi8>, %arg4: tensor<81x14xi64>, %arg5: tensor<81x1xi64>, %arg6: tensor<98x79x24xf32>) -> (tensor<81x14xi1>, tensor<1xi1>, tensor<98x79x24xf32>, tensor<31x83x59x43x85x70xi1>, tensor<31x83x59x43x85x70xi1>, tensor<98x79x24xf32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<61xf32>, tensor<1xf32>) -> tensor<61xi1>
    %1 = tosa.equal %arg2, %arg3 : (tensor<31x83x59x43x85x70xi8>, tensor<31x1x1x43x1x70xi8>) -> tensor<31x83x59x43x85x70xi1>
    %2 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<61xi1>) -> tensor<1xi1>
    %3 = tosa.greater_equal %arg4, %arg5 : (tensor<81x14xi64>, tensor<81x1xi64>) -> tensor<81x14xi1>
    %4 = tosa.reverse %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.identity %1 : (tensor<31x83x59x43x85x70xi1>) -> tensor<31x83x59x43x85x70xi1>
    %6 = tosa.rsqrt %arg6 : (tensor<98x79x24xf32>) -> tensor<98x79x24xf32>
    %7 = tosa.bitwise_or %5, %5 : (tensor<31x83x59x43x85x70xi1>, tensor<31x83x59x43x85x70xi1>) -> tensor<31x83x59x43x85x70xi1>
    %8 = tosa.maximum %6, %6 : (tensor<98x79x24xf32>, tensor<98x79x24xf32>) -> tensor<98x79x24xf32>
    %9 = tosa.bitwise_xor %7, %7 : (tensor<31x83x59x43x85x70xi1>, tensor<31x83x59x43x85x70xi1>) -> tensor<31x83x59x43x85x70xi1>
    %10 = tosa.logical_and %5, %7 : (tensor<31x83x59x43x85x70xi1>, tensor<31x83x59x43x85x70xi1>) -> tensor<31x83x59x43x85x70xi1>
    %11 = tosa.sigmoid %6 : (tensor<98x79x24xf32>) -> tensor<98x79x24xf32>
    return %3, %4, %8, %9, %10, %11 : tensor<81x14xi1>, tensor<1xi1>, tensor<98x79x24xf32>, tensor<31x83x59x43x85x70xi1>, tensor<31x83x59x43x85x70xi1>, tensor<98x79x24xf32>
  }
}
