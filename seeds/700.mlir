module {
  func.func @main(%arg0: tensor<86xi64>, %arg1: tensor<1xi64>, %arg2: tensor<50xf32>, %arg3: tensor<50xf32>) -> (tensor<50xf32>, tensor<86xi1>, tensor<86xi1>, tensor<86xi1>, tensor<50xf32>, tensor<i32>, tensor<i32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<86xi64>, tensor<1xi64>) -> tensor<86xi1>
    %1 = tosa.pow %arg2, %arg3 : (tensor<50xf32>, tensor<50xf32>) -> tensor<50xf32>
    %2 = tosa.log %1 : (tensor<50xf32>) -> tensor<50xf32>
    %3 = tosa.argmax %1 {axis = 0 : i32} : (tensor<50xf32>) -> tensor<i32>
    %4 = tosa.intdiv %3, %3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = tosa.logical_and %0, %0 : (tensor<86xi1>, tensor<86xi1>) -> tensor<86xi1>
    %6 = tosa.logical_and %5, %5 : (tensor<86xi1>, tensor<86xi1>) -> tensor<86xi1>
    %7 = tosa.minimum %2, %2 : (tensor<50xf32>, tensor<50xf32>) -> tensor<50xf32>
    %8 = tosa.reverse %6 {axis = 0 : i32} : (tensor<86xi1>) -> tensor<86xi1>
    %9 = tosa.reverse %6 {axis = 0 : i32} : (tensor<86xi1>) -> tensor<86xi1>
    %10 = tosa.tanh %1 : (tensor<50xf32>) -> tensor<50xf32>
    %11 = tosa.logical_not %0 : (tensor<86xi1>) -> tensor<86xi1>
    %12 = tosa.intdiv %3, %3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %13 = tosa.maximum %10, %10 : (tensor<50xf32>, tensor<50xf32>) -> tensor<50xf32>
    %14 = tosa.abs %12 : (tensor<i32>) -> tensor<i32>
    %15 = tosa.abs %14 : (tensor<i32>) -> tensor<i32>
    %16 = tosa.arithmetic_right_shift %4, %4 {round = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %7, %8, %9, %11, %13, %15, %16 : tensor<50xf32>, tensor<86xi1>, tensor<86xi1>, tensor<86xi1>, tensor<50xf32>, tensor<i32>, tensor<i32>
  }
}
