module {
  func.func @main(%arg0: tensor<55x71x23x69xi64>, %arg1: tensor<1x71x23x1xi64>, %arg2: tensor<86xf32>) -> (tensor<55x1x23x69xi1>, tensor<86xf32>, tensor<55x1x23x69xi1>, tensor<55x71x23x69xi1>, tensor<86xf32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<55x71x23x69xi64>, tensor<1x71x23x1xi64>) -> tensor<55x71x23x69xi1>
    %1 = tosa.reciprocal %arg2 : (tensor<86xf32>) -> tensor<86xf32>
    %2 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<55x71x23x69xi1>) -> tensor<55x1x23x69xi1>
    %3 = tosa.logical_xor %2, %2 : (tensor<55x1x23x69xi1>, tensor<55x1x23x69xi1>) -> tensor<55x1x23x69xi1>
    %4 = tosa.abs %1 : (tensor<86xf32>) -> tensor<86xf32>
    %5 = tosa.log %4 : (tensor<86xf32>) -> tensor<86xf32>
    %6 = tosa.clz %2 : (tensor<55x1x23x69xi1>) -> tensor<55x1x23x69xi1>
    %7 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<55x71x23x69xi1>, tensor<55x71x23x69xi1>) -> tensor<55x71x23x69xi1>
    %8 = tosa.reverse %4 {axis = 0 : i32} : (tensor<86xf32>) -> tensor<86xf32>
    return %3, %5, %6, %7, %8 : tensor<55x1x23x69xi1>, tensor<86xf32>, tensor<55x1x23x69xi1>, tensor<55x71x23x69xi1>, tensor<86xf32>
  }
}
