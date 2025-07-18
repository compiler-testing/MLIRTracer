module {
  func.func @main(%arg0: tensor<7x66xf32>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<31x29x22x37x92x73xi1>, %arg4: tensor<2xi32>, %arg5: tensor<2xi32>) -> (tensor<31x29x22x37x92x73xi1>, tensor<i64>, tensor<i32>, tensor<2xi32>, tensor<7x66xi1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<7x66xf32>) -> tensor<7x66xf32>
    %1 = tosa.bitwise_xor %arg1, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = tosa.sigmoid %0 : (tensor<7x66xf32>) -> tensor<7x66xf32>
    %3 = tosa.logical_not %arg3 : (tensor<31x29x22x37x92x73xi1>) -> tensor<31x29x22x37x92x73xi1>
    %4 = tosa.bitwise_and %1, %1 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %5 = tosa.intdiv %arg4, %arg5 : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    %6 = tosa.sigmoid %2 : (tensor<7x66xf32>) -> tensor<7x66xf32>
    %7 = tosa.clz %5 : (tensor<2xi32>) -> tensor<2xi32>
    %8 = tosa.minimum %7, %7 : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    %9 = tosa.argmax %5 {axis = 0 : i32} : (tensor<2xi32>) -> tensor<i32>
    %10 = tosa.abs %6 : (tensor<7x66xf32>) -> tensor<7x66xf32>
    %11 = tosa.logical_right_shift %8, %7 : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    %12 = tosa.equal %10, %6 : (tensor<7x66xf32>, tensor<7x66xf32>) -> tensor<7x66xi1>
    return %3, %4, %9, %11, %12 : tensor<31x29x22x37x92x73xi1>, tensor<i64>, tensor<i32>, tensor<2xi32>, tensor<7x66xi1>
  }
}
