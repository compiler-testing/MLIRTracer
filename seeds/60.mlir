module {
  func.func @main(%arg0: tensor<84x44x79x99xi64>, %arg1: tensor<33x45x79x5xi1>, %arg2: tensor<65x56xf32>) -> (tensor<1x45x79x5xi1>, tensor<84x44x79x99xi64>, tensor<65x56xf32>, tensor<65x1xi1>) {
    %0 = tosa.abs %arg0 : (tensor<84x44x79x99xi64>) -> tensor<84x44x79x99xi64>
    %1 = tosa.identity %0 : (tensor<84x44x79x99xi64>) -> tensor<84x44x79x99xi64>
    %2 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<33x45x79x5xi1>) -> tensor<1x45x79x5xi1>
    %3 = tosa.add %1, %0 : (tensor<84x44x79x99xi64>, tensor<84x44x79x99xi64>) -> tensor<84x44x79x99xi64>
    %4 = tosa.reciprocal %arg2 : (tensor<65x56xf32>) -> tensor<65x56xf32>
    %5 = tosa.bitwise_xor %3, %3 : (tensor<84x44x79x99xi64>, tensor<84x44x79x99xi64>) -> tensor<84x44x79x99xi64>
    %6 = tosa.greater_equal %4, %4 : (tensor<65x56xf32>, tensor<65x56xf32>) -> tensor<65x56xi1>
    %7 = tosa.floor %4 : (tensor<65x56xf32>) -> tensor<65x56xf32>
    %8 = tosa.reduce_min %6 {axis = 1 : i32} : (tensor<65x56xi1>) -> tensor<65x1xi1>
    %9 = tosa.logical_or %8, %8 : (tensor<65x1xi1>, tensor<65x1xi1>) -> tensor<65x1xi1>
    return %2, %5, %7, %9 : tensor<1x45x79x5xi1>, tensor<84x44x79x99xi64>, tensor<65x56xf32>, tensor<65x1xi1>
  }
}
