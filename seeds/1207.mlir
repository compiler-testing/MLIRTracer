module {
  func.func @main(%arg0: tensor<60xi32>, %arg1: tensor<65x89x66x43x31xf32>, %arg2: tensor<52xi1>) -> (tensor<65x89x66x43x31xf32>, tensor<65x89x66x43x31xf32>, tensor<1xi1>, tensor<60xi32>, tensor<65x89x66x86x31xi1>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<60xi32>) -> tensor<60xi32>
    %1 = tosa.rsqrt %arg1 : (tensor<65x89x66x43x31xf32>) -> tensor<65x89x66x43x31xf32>
    %2 = tosa.minimum %1, %1 : (tensor<65x89x66x43x31xf32>, tensor<65x89x66x43x31xf32>) -> tensor<65x89x66x43x31xf32>
    %3 = tosa.equal %2, %2 : (tensor<65x89x66x43x31xf32>, tensor<65x89x66x43x31xf32>) -> tensor<65x89x66x43x31xi1>
    %4 = tosa.reciprocal %1 : (tensor<65x89x66x43x31xf32>) -> tensor<65x89x66x43x31xf32>
    %5 = tosa.arithmetic_right_shift %3, %3 {round = true} : (tensor<65x89x66x43x31xi1>, tensor<65x89x66x43x31xi1>) -> tensor<65x89x66x43x31xi1>
    %6 = tosa.clz %5 : (tensor<65x89x66x43x31xi1>) -> tensor<65x89x66x43x31xi1>
    %7 = tosa.floor %1 : (tensor<65x89x66x43x31xf32>) -> tensor<65x89x66x43x31xf32>
    %8 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<52xi1>) -> tensor<1xi1>
    %t_9 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %9 = tosa.tile %0, %t_9 : (tensor<60xi32>, !tosa.shape<1>) -> tensor<60xi32>
    %10 = tosa.bitwise_not %9 : (tensor<60xi32>) -> tensor<60xi32>
    %11 = tosa.logical_right_shift %6, %5 : (tensor<65x89x66x43x31xi1>, tensor<65x89x66x43x31xi1>) -> tensor<65x89x66x43x31xi1>
    %12 = tosa.bitwise_and %10, %10 : (tensor<60xi32>, tensor<60xi32>) -> tensor<60xi32>
    %13 = tosa.concat %11, %6 {axis = 3 : i32} : (tensor<65x89x66x43x31xi1>, tensor<65x89x66x43x31xi1>) -> tensor<65x89x66x86x31xi1>
    return %4, %7, %8, %12, %13 : tensor<65x89x66x43x31xf32>, tensor<65x89x66x43x31xf32>, tensor<1xi1>, tensor<60xi32>, tensor<65x89x66x86x31xi1>
  }
}
