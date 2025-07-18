module {
  func.func @main(%arg0: tensor<52x89x52x54xi32>, %arg1: tensor<1x89x52x1xi32>) -> tensor<52x89x52x1xi1> {
    %0 = tosa.equal %arg0, %arg1 : (tensor<52x89x52x54xi32>, tensor<1x89x52x1xi32>) -> tensor<52x89x52x54xi1>
    %1 = tosa.abs %0 : (tensor<52x89x52x54xi1>) -> tensor<52x89x52x54xi1>
    %2 = tosa.bitwise_or %1, %1 : (tensor<52x89x52x54xi1>, tensor<52x89x52x54xi1>) -> tensor<52x89x52x54xi1>
    %3 = tosa.abs %2 : (tensor<52x89x52x54xi1>) -> tensor<52x89x52x54xi1>
    %4 = tosa.bitwise_or %3, %0 : (tensor<52x89x52x54xi1>, tensor<52x89x52x54xi1>) -> tensor<52x89x52x54xi1>
    %5 = tosa.bitwise_and %4, %3 : (tensor<52x89x52x54xi1>, tensor<52x89x52x54xi1>) -> tensor<52x89x52x54xi1>
    %6 = tosa.reduce_sum %5 {axis = 3 : i32} : (tensor<52x89x52x54xi1>) -> tensor<52x89x52x1xi1>
    %7 = tosa.logical_not %6 : (tensor<52x89x52x1xi1>) -> tensor<52x89x52x1xi1>
    return %7 : tensor<52x89x52x1xi1>
  }
}
